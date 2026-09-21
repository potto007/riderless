#include "chat.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include "worker-utils.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using steady_clock = std::chrono::steady_clock;

namespace {

constexpr size_t MAX_PROTOCOL_BYTES = 4 * 1024 * 1024;
constexpr const char * MODEL_ID = "local-gemma-riderless-v1";
constexpr const char * PROMPT_VERSION = "riderless-gemma-choice-v1";
constexpr const char * ANSWER_PREFIX = "Answer:\n";
constexpr const char * LABEL_CANDIDATES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
// Splitting a prefill costs one extra decode (about 13 ms measured on the RTX
// 5090), which a shorter shared prefix does not repay across a few questions.
constexpr size_t MIN_SHARED_PREFIX_TOKENS = 128;

struct settings {
    std::string model_path;
    std::string model_sha256;
    std::string runtime_sha256;
    int context = 2048;
    int batch = 256;
    int ubatch = 256;
    int threads = 8;
    int max_questions = 32;
    bool gpu = false;
};

struct prepared_question {
    std::string id;
    std::string prompt;
    std::vector<llama_token> tokens;
    size_t prefix_tokens = 0;
    std::vector<std::string> labels;
    std::vector<int> label_token_ids;
};

double elapsed_ms(const steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
        steady_clock::now() - started).count();
}

int parse_positive(const std::string & text, const std::string & name) {
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::runtime_error(name + " must be a positive integer");
    }
    return value;
}

settings parse_args(int argc, char ** argv) {
    std::map<std::string, std::string> values;
    bool gpu = false;
    const std::set<std::string> valued{
        "--model", "--model-sha256", "--runtime-sha256", "--context",
        "--batch", "--ubatch", "--threads", "--max-questions",
    };
    for (int index = 1; index < argc; ++index) {
        const std::string key(argv[index]);
        if (key == "--gpu") {
            if (gpu) throw std::runtime_error("Repeated --gpu");
            gpu = true;
            continue;
        }
        if (!valued.count(key) || index + 1 >= argc || values.count(key)) {
            throw std::runtime_error("Invalid or repeated argument: " + key);
        }
        values[key] = argv[++index];
    }
    for (const auto * key : {
             "--model", "--model-sha256", "--runtime-sha256", "--context",
             "--batch", "--ubatch", "--threads", "--max-questions"}) {
        if (!values.count(key)) throw std::runtime_error(std::string("Missing ") + key);
    }
    settings result;
    result.model_path = values.at("--model");
    result.model_sha256 = values.at("--model-sha256");
    result.runtime_sha256 = values.at("--runtime-sha256");
    result.context = parse_positive(values.at("--context"), "context");
    result.batch = parse_positive(values.at("--batch"), "batch");
    result.ubatch = parse_positive(values.at("--ubatch"), "ubatch");
    result.threads = parse_positive(values.at("--threads"), "threads");
    result.max_questions = parse_positive(
        values.at("--max-questions"), "max-questions");
    result.gpu = gpu;
    if (result.max_questions > 32) throw std::runtime_error("max-questions exceeds 32");
    return result;
}

void reject_incompatible_environment() {
    for (const auto * name : {
             "RIDERLESS_GEMMA4_EXIT_LAYER",
             "GGML_CUDA_DISABLE_FUSION",
             "GGML_CUDA_DISABLE_GRAPHS"}) {
        if (std::getenv(name)) {
            throw std::runtime_error(std::string("Incompatible environment setting: ") + name);
        }
    }
}

std::vector<llama_token> tokenize(
        const llama_vocab * vocab, const std::string & text, bool add_special = true) {
    const int count = llama_tokenize(
        vocab, text.data(), text.size(), nullptr, 0, add_special, true);
    if (count >= 0) throw std::runtime_error("Empty or invalid tokenization");
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    const int written = llama_tokenize(
        vocab, text.data(), text.size(), tokens.data(), tokens.size(),
        add_special, true);
    if (written != static_cast<int>(tokens.size())) {
        throw std::runtime_error("Tokenization size changed");
    }
    return tokens;
}

struct control_token_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct budget_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The rendered prompt is tokenized with special-token parsing so the template's
// own turn markers work. Caller text must therefore never spell a control token,
// or it could forge a turn boundary ahead of the readout position.
void reject_control_tokens(const llama_vocab * vocab, const json & messages) {
    for (const auto & message : messages) {
        const std::string content = message.at("content").get<std::string>();
        if (content.empty()) continue;
        for (const llama_token token : tokenize(vocab, content, false)) {
            if (llama_vocab_is_control(vocab, token) || llama_vocab_is_eog(vocab, token)) {
                throw control_token_error("Caller content contains a control token");
            }
        }
    }
}

void require_single_bos(
        const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    const llama_token bos = llama_vocab_bos(vocab);
    if (tokens.size() > 1 && tokens.at(0) == bos && tokens.at(1) == bos) {
        throw std::runtime_error("Rendered prompt starts with a doubled BOS");
    }
}

std::string render_prompt(
        const json & messages,
        const common_chat_templates * templates,
        const std::string & answer_prefix) {
    common_chat_templates_inputs inputs;
    // llama.cpp v0.4.1 moved the common library off nlohmann and onto its own
    // common_json. The protocol stays nlohmann, so the already-validated message
    // array is handed over verbatim through its serialization: the template sees
    // exactly the messages it saw before.
    inputs.messages = common_chat_msgs_parse_oaicompat(
        common_json::parse(messages.dump()));
    inputs.enable_thinking = false;
    inputs.chat_template_kwargs["enable_thinking"] = "false";
    inputs.now = std::chrono::system_clock::time_point{};
    return common_chat_templates_apply(templates, inputs).prompt + answer_prefix;
}

std::string sha256_hex(const std::string & value) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(
        digest,
        reinterpret_cast<const unsigned char *>(value.data()),
        value.size());
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

bool read_bounded_line(std::istream & stream, std::string & line, size_t limit) {
    line.clear();
    char value = '\0';
    while (stream.get(value)) {
        if (value == '\n') return true;
        if (line.size() == limit) {
            throw std::runtime_error("Protocol request exceeds size limit");
        }
        line.push_back(value);
    }
    return !line.empty();
}

// Where the caller-marked shared text ends, in tokens of this question's own
// prompt. It depends on nothing but this prompt, so a question is computed the
// same way alone, reordered, or beside any siblings. It is only a hint: any
// value below the prompt length is correct, a poor one just reuses less.
size_t shared_prefix_tokens(
        const json & row,
        const llama_vocab * vocab,
        const prepared_question & prepared) {
    const size_t bytes = row.value("shared_prefix_bytes", size_t{0});
    if (bytes == 0) return 0;
    const std::string content =
        row.at("messages").at(0).at("content").get<std::string>();
    if (bytes > content.size()) {
        throw std::runtime_error("Shared prefix exceeds message content");
    }
    const size_t start = prepared.prompt.find(content.substr(0, bytes));
    if (start == std::string::npos) return 0;
    const auto head = tokenize(vocab, prepared.prompt.substr(0, start + bytes));
    // The last head token can merge with whatever follows it, so it never counts.
    const size_t limit = std::min(head.size(), prepared.tokens.size()) - 1;
    size_t common = 0;
    while (common < limit && head.at(common) == prepared.tokens.at(common)) ++common;
    return common < MIN_SHARED_PREFIX_TOKENS ? 0 : common;
}

prepared_question prepare_question(
        const json & row,
        const llama_vocab * vocab,
        const common_chat_templates * templates,
        const std::vector<std::string> & supported_labels,
        const std::vector<int> & supported_token_ids,
        int context_size) {
    prepared_question prepared;
    prepared.id = row.at("id").get<std::string>();
    if (prepared.id.empty()) throw std::runtime_error("Empty question id");
    if (row.at("prompt_version").get<std::string>() != PROMPT_VERSION) {
        throw std::runtime_error("Unsupported prompt version");
    }
    const std::string answer_prefix = row.at("answer_prefix").get<std::string>();
    if (answer_prefix != ANSWER_PREFIX) {
        throw std::runtime_error("Unsupported answer prefix");
    }
    prepared.labels = row.at("labels").get<std::vector<std::string>>();
    if (prepared.labels.size() < 2 || prepared.labels.size() > supported_labels.size()) {
        throw std::runtime_error("Unsupported label cardinality");
    }
    if (!std::equal(
            prepared.labels.begin(), prepared.labels.end(), supported_labels.begin())) {
        throw std::runtime_error("Labels differ from validated alphabet prefix");
    }
    reject_control_tokens(vocab, row.at("messages"));
    prepared.prompt = render_prompt(row.at("messages"), templates, answer_prefix);
    prepared.tokens = tokenize(vocab, prepared.prompt);
    require_single_bos(vocab, prepared.tokens);
    if (prepared.tokens.size() > static_cast<size_t>(context_size)) {
        throw budget_error("Prompt exceeds context size");
    }
    prepared.prefix_tokens = shared_prefix_tokens(row, vocab, prepared);
    for (size_t index = 0; index < prepared.labels.size(); ++index) {
        const auto extended = tokenize(vocab, prepared.prompt + prepared.labels.at(index));
        const int token_id = riderless::single_suffix_token(prepared.tokens, extended);
        if (token_id != supported_token_ids.at(index)) {
            throw std::runtime_error("Prompt-specific label token mapping changed");
        }
        prepared.label_token_ids.push_back(token_id);
    }
    return prepared;
}

std::pair<std::vector<std::string>, std::vector<int>> validate_alphabet(
        const llama_vocab * vocab,
        const common_chat_templates * templates) {
    const json messages = json::array({
        {{"role", "user"}, {"content",
            "Choose one option.\n\nOPTIONS:\nA: first\nB: second\n\n"
            "Reply with one option label only."}},
    });
    const std::string prompt = render_prompt(messages, templates, ANSWER_PREFIX);
    const auto prompt_tokens = tokenize(vocab, prompt);
    require_single_bos(vocab, prompt_tokens);
    std::vector<std::string> labels;
    std::vector<int> token_ids;
    for (const char candidate : std::string(LABEL_CANDIDATES)) {
        const std::string label(1, candidate);
        try {
            const auto extended = tokenize(vocab, prompt + label);
            const int token_id = riderless::single_suffix_token(prompt_tokens, extended);
            if (std::find(token_ids.begin(), token_ids.end(), token_id) != token_ids.end()) {
                break;
            }
            labels.push_back(label);
            token_ids.push_back(token_id);
        } catch (const std::runtime_error &) {
            break;
        }
    }
    if (labels.size() < 10) {
        throw std::runtime_error("Validated label alphabet is too small");
    }
    return {labels, token_ids};
}

json evaluate_question(
        llama_context * context,
        const llama_vocab * vocab,
        const prepared_question & question,
        int batch_size,
        std::vector<llama_token> & cached_prefix) {
    const auto started = steady_clock::now();
    const size_t prefix = question.prefix_tokens;
    // Reuse needs the cached tokens to equal this question's prefix exactly. A
    // longer or shorter match is recomputed, so the prefix is always prefilled
    // in the same chunks and the result does not depend on the siblings.
    const bool reuse = prefix > 0 && cached_prefix.size() == prefix &&
        std::equal(cached_prefix.begin(), cached_prefix.end(), question.tokens.begin());
    auto * memory = llama_get_memory(context);
    if (reuse) {
        if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(prefix), -1)) {
            throw std::runtime_error("Context memory could not be trimmed to the prefix");
        }
    } else {
        llama_memory_clear(memory, true);
        cached_prefix.clear();
    }
    // cache_cleared and reused_tokens below are measured facts: sequence 0 must
    // hold exactly the reused prefix, or nothing.
    const llama_pos held = reuse ? static_cast<llama_pos>(prefix) - 1 : -1;
    if (llama_memory_seq_pos_max(memory, 0) != held) {
        throw std::runtime_error("Context memory differs from the expected prefix");
    }
    size_t processed = 0;
    const auto prefill = [&](size_t begin, size_t end) {
        size_t offset = begin;
        for (const size_t count : riderless::prefill_chunks(
                 end - begin, static_cast<size_t>(batch_size))) {
            auto batch = llama_batch_get_one(
                const_cast<llama_token *>(question.tokens.data()) + offset,
                static_cast<int32_t>(count));
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("Full-model prefill failed");
            }
            offset += count;
            processed += count;
        }
    };
    if (!reuse && prefix > 0) {
        prefill(0, prefix);
        cached_prefix.assign(
            question.tokens.begin(),
            question.tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
    }
    prefill(prefix, question.tokens.size());
    llama_synchronize(context);
    const float * raw_logits = llama_get_logits_ith(context, -1);
    if (!raw_logits) throw std::runtime_error("Missing full-model logits");
    const int vocabulary_size = llama_vocab_n_tokens(vocab);
    std::vector<double> vocabulary_logits;
    vocabulary_logits.reserve(static_cast<size_t>(vocabulary_size));
    for (int token_id = 0; token_id < vocabulary_size; ++token_id) {
        vocabulary_logits.push_back(raw_logits[token_id]);
    }
    std::vector<double> label_logits;
    for (const int token_id : question.label_token_ids) {
        label_logits.push_back(vocabulary_logits.at(static_cast<size_t>(token_id)));
    }
    (void) riderless::stable_softmax(label_logits);
    const auto summary = riderless::summarize_vocabulary(
        vocabulary_logits, question.label_token_ids);
    const size_t reused = reuse ? prefix : 0;
    if (processed + reused != question.tokens.size()) {
        throw std::runtime_error("Processed token accounting changed");
    }
    return {
        {"id", question.id},
        {"label_logits", label_logits},
        {"label_token_ids", question.label_token_ids},
        {"allowed_label_mass", summary.allowed_mass},
        {"full_vocabulary_argmax", {
            {"token_id", summary.argmax_token_id},
            {"logit", summary.argmax_logit},
        }},
        {"prompt_sha256", sha256_hex(question.prompt)},
        {"prompt_tokens", question.tokens.size()},
        {"processed_tokens", processed},
        {"reused_tokens", reused},
        {"cache_cleared", !reuse},
        {"timing_ms", elapsed_ms(started)},
    };
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const settings config = parse_args(argc, argv);
        reject_incompatible_environment();
        ggml_backend_load_all_from_path(RIDERLESS_BACKEND_DIR);
        if (config.gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            throw std::runtime_error("GPU opt-in requested but no GPU backend is available");
        }
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.gpu ? 999 : 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(config.model_path.c_str(), model_params),
            llama_model_free);
        if (!model) throw std::runtime_error("Model load failed");
        char architecture[64];
        if (llama_model_meta_val_str(
                model.get(), "general.architecture", architecture,
                sizeof(architecture)) < 1 || std::string(architecture) != "gemma4") {
            throw std::runtime_error("Worker supports Gemma4 only");
        }
        char description[256];
        if (llama_model_desc(model.get(), description, sizeof(description)) < 1) {
            throw std::runtime_error("Model identity is unavailable");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        auto templates = common_chat_templates_init(model.get(), "");
        auto [labels, token_ids] = validate_alphabet(vocab, templates.get());

        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(config.context);
        context_params.n_batch = static_cast<uint32_t>(config.batch);
        context_params.n_ubatch = static_cast<uint32_t>(config.ubatch);
        context_params.n_seq_max = 1;
        context_params.n_threads = config.threads;
        context_params.n_threads_batch = config.threads;
        context_params.swa_full = true;
        context_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        context_params.offload_kqv = config.gpu;
        context_params.op_offload = config.gpu;
        context_params.no_perf = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("Context initialization failed");

        const json hello = {
            {"type", "hello"},
            {"protocol_version", 2},
            {"model_id", MODEL_ID},
            {"model_name", std::string(description)},
            {"model_sha256", config.model_sha256},
            {"runtime_sha256", config.runtime_sha256},
            {"labels", labels},
            {"label_token_ids", token_ids},
            {"context_size", config.context},
            {"batch_size", config.batch},
            {"ubatch_size", config.ubatch},
            {"threads", config.threads},
            {"max_questions", config.max_questions},
            {"generated_tokens", 0},
            {"callbacks_enabled", false},
            {"execution_mode", "full"},
        };
        std::cout << hello.dump() << '\n' << std::flush;

        std::string line;
        while (read_bounded_line(std::cin, line, MAX_PROTOCOL_BYTES)) {
            if (line.empty()) continue;
            std::string correlation_id = "unknown";
            std::vector<prepared_question> questions;
            try {
                const json request = json::parse(line);
                correlation_id = request.at("id").get<std::string>();
                if (correlation_id.empty() || request.at("type") != "evaluate") {
                    throw std::runtime_error("Invalid request envelope");
                }
                const auto & rows = request.at("questions");
                if (!rows.is_array() || rows.empty() ||
                        rows.size() > static_cast<size_t>(config.max_questions)) {
                    throw std::runtime_error("Invalid question count");
                }
                std::set<std::string> ids;
                for (const auto & row : rows) {
                    auto prepared = prepare_question(
                        row, vocab, templates.get(), labels, token_ids, config.context);
                    if (!ids.insert(prepared.id).second) {
                        throw std::runtime_error("Duplicate question id");
                    }
                    questions.push_back(std::move(prepared));
                }
            } catch (const std::exception & error) {
                std::cerr << "PREFLIGHT_FAILED " << error.what() << '\n';
                const bool control =
                    dynamic_cast<const control_token_error *>(&error) != nullptr;
                const bool budget =
                    dynamic_cast<const budget_error *>(&error) != nullptr;
                const json failure = {
                    {"type", "error"},
                    {"id", correlation_id},
                    {"code", "invalid_request"},
                    {"reason", control ? "control_tokens" : budget ? "budget" : "internal"},
                };
                std::cout << failure.dump() << '\n' << std::flush;
                continue;
            }
            json results = json::array();
            // Never outlives the request: no state is carried between callers.
            std::vector<llama_token> cached_prefix;
            for (const auto & question : questions) {
                results.push_back(evaluate_question(
                    context.get(), vocab, question, config.batch, cached_prefix));
            }
            const json response = {
                {"type", "result"},
                {"id", correlation_id},
                {"model_sha256", config.model_sha256},
                {"runtime_sha256", config.runtime_sha256},
                {"generated_tokens", 0},
                {"callbacks_enabled", false},
                {"execution_mode", "full"},
                {"questions", results},
            };
            const std::string serialized = response.dump();
            if (serialized.size() > MAX_PROTOCOL_BYTES) {
                throw std::runtime_error("Protocol response exceeds size limit");
            }
            std::cout << serialized << '\n' << std::flush;
        }
        context.reset();
        templates.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "WORKER_FAILED " << error.what() << '\n';
        return 1;
    }
}
