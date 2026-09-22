// A logits server for competitor benchmarks, on riderless's own runtime.
//
// The Open Alternative to Jev ("so1") backend contract asks for the score of
// each label token at arbitrary readout positions of arbitrary token
// sequences. Its shipped backends are transformers and vLLM, neither of which
// can load the GGUF riderless is measured on. This helper closes that gap: it
// loads the same file with the same llama_model_params and the same
// llama_context_params worker.cpp uses, so a so1 run and a riderless run
// differ only in the prompt bytes and the readout position, never in the
// weights, the engine or the kernels.
//
// It is deliberately not the worker. It renders no chat template, compiles no
// question and hashes no prompt: the caller supplies token ids that the
// competitor's own prompt builder produced. The only thing shared is how those
// ids are put through the model.
//
// Protocol, one JSON object per line on stdin, one per line on stdout:
//   {"type":"tokenize","id":S,"text":S,"add_special":B}
//       -> {"type":"tokenize","id":S,"ids":[I]}
//   {"type":"scores","id":S,"label_ids":[I],
//    "sequences":[{"ids":[I],"positions":[I]}]}
//       -> {"type":"scores","id":S,"scores":[[[F]]],"readouts":[[O]],
//           "prompt_tokens":[I],"decode_ms":[F],"total_ms":F}
// Each readout object carries the same two facts riderless's own diagnostics
// carry at its readout position, `allowed_label_mass` and
// `full_vocabulary_argmax`, so a competitor row can be read beside a riderless
// row without inventing them.
// A failed request answers {"type":"error","id":S,"error":S} and the helper
// keeps serving, so one bad case is evidence rather than a lost run.

#include "llama.h"
#include "nlohmann/json.hpp"
#include "worker-utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using steady_clock = std::chrono::steady_clock;

namespace {

constexpr size_t MAX_PROTOCOL_BYTES = 32 * 1024 * 1024;
constexpr int PROTOCOL_VERSION = 1;

struct settings {
    std::string model_path;
    // worker.cpp's sequential defaults. Only --context is expected to move,
    // because a packed competitor prompt is longer than a riderless one.
    int context = 2048;
    int batch = 256;
    int ubatch = 256;
    int threads = 8;
    bool gpu = false;
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
    const std::set<std::string> valued{
        "--model", "--context", "--batch", "--ubatch", "--threads"};
    std::map<std::string, std::string> values{
        {"--context", "2048"}, {"--batch", "256"},
        {"--ubatch", "256"}, {"--threads", "8"}};
    bool gpu = false;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "--gpu") {
            gpu = true;
            continue;
        }
        if (valued.count(flag) == 0 || index + 1 >= argc) {
            throw std::runtime_error("Unknown or incomplete argument: " + flag);
        }
        values[flag] = argv[++index];
    }
    if (values.count("--model") == 0) {
        throw std::runtime_error("--model is required");
    }
    settings result;
    result.model_path = values.at("--model");
    result.context = parse_positive(values.at("--context"), "context");
    result.batch = parse_positive(values.at("--batch"), "batch");
    result.ubatch = parse_positive(values.at("--ubatch"), "ubatch");
    result.threads = parse_positive(values.at("--threads"), "threads");
    result.gpu = gpu;
    return result;
}

// worker.cpp refuses these three, because each one changes the kernels the
// published measurements were taken with. The same refusal here keeps a so1
// row comparable to a riderless row.
void reject_incompatible_environment() {
    for (const auto * name : {
             "RIDERLESS_GEMMA4_EXIT_LAYER",
             "GGML_CUDA_DISABLE_FUSION",
             "GGML_CUDA_DISABLE_GRAPHS"}) {
        if (std::getenv(name)) {
            throw std::runtime_error(
                std::string("Incompatible environment setting: ") + name);
        }
    }
}

// worker.cpp:221, kept local because it lives in that file's anonymous
// namespace and the worker is not this helper's to edit.
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

std::vector<llama_token> tokenize(
        const llama_vocab * vocab, const std::string & text, bool add_special) {
    if (text.empty()) {
        return {};
    }
    const int count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
        add_special, true);
    if (count >= 0) throw std::runtime_error("Empty or invalid tokenization");
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    const int written = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), add_special, true);
    if (written != static_cast<int>(tokens.size())) {
        throw std::runtime_error("Tokenization size changed");
    }
    return tokens;
}

struct sequence_request {
    std::vector<llama_token> ids;
    std::vector<size_t> positions;
};

// A heap batch, because the readout needs the logits flag set at chosen
// positions and llama_batch_get_one only ever marks the last token.
class batch_holder {
public:
    explicit batch_holder(int32_t capacity)
        : batch(llama_batch_init(capacity, 0, 1)) {}
    ~batch_holder() { llama_batch_free(batch); }
    batch_holder(const batch_holder &) = delete;
    batch_holder & operator=(const batch_holder &) = delete;

    llama_batch batch;
};

std::vector<sequence_request> parse_sequences(const json & request, int context) {
    const auto & rows = request.at("sequences");
    if (!rows.is_array() || rows.empty()) {
        throw std::runtime_error("sequences must be a nonempty array");
    }
    std::vector<sequence_request> sequences;
    for (const auto & row : rows) {
        sequence_request sequence;
        sequence.ids = row.at("ids").get<std::vector<llama_token>>();
        if (sequence.ids.empty()) {
            throw std::runtime_error("a sequence has no tokens");
        }
        if (sequence.ids.size() > static_cast<size_t>(context)) {
            throw std::runtime_error(
                "sequence of " + std::to_string(sequence.ids.size()) +
                " tokens exceeds the context of " + std::to_string(context));
        }
        sequence.positions = row.at("positions").get<std::vector<size_t>>();
        if (sequence.positions.empty()) {
            throw std::runtime_error("a sequence has no readout positions");
        }
        size_t previous = 0;
        for (size_t order = 0; order < sequence.positions.size(); ++order) {
            const size_t position = sequence.positions.at(order);
            if (position >= sequence.ids.size()) {
                throw std::runtime_error("readout position is past the sequence");
            }
            if (order > 0 && position <= previous) {
                throw std::runtime_error("readout positions must ascend");
            }
            previous = position;
        }
        sequences.push_back(std::move(sequence));
    }
    return sequences;
}

// One sequence, cleared cache, chunked prefill, label logits at each readout.
json score_sequence(
        llama_context * context,
        const llama_vocab * vocab,
        const sequence_request & sequence,
        const std::vector<int32_t> & label_ids,
        int batch_size,
        batch_holder & holder,
        json & readouts,
        double & decode_ms) {
    const auto started = steady_clock::now();
    llama_memory_clear(llama_get_memory(context), true);
    const int vocabulary_size = llama_vocab_n_tokens(vocab);
    for (const int32_t token_id : label_ids) {
        if (token_id < 0 || token_id >= vocabulary_size) {
            throw std::runtime_error("label token id is outside the vocabulary");
        }
    }

    json rows = json::array();
    size_t offset = 0;
    size_t next_readout = 0;
    for (const size_t count : riderless::prefill_chunks(
             sequence.ids.size(), static_cast<size_t>(batch_size))) {
        llama_batch & batch = holder.batch;
        batch.n_tokens = static_cast<int32_t>(count);
        // Which slots of this chunk the caller wants logits for, paired with
        // their place in the chunk so llama_get_logits_ith can find them.
        std::vector<int32_t> wanted;
        for (size_t local = 0; local < count; ++local) {
            const size_t absolute = offset + local;
            batch.token[local] = sequence.ids.at(absolute);
            batch.pos[local] = static_cast<llama_pos>(absolute);
            batch.n_seq_id[local] = 1;
            batch.seq_id[local][0] = 0;
            const bool readout = next_readout < sequence.positions.size() &&
                sequence.positions.at(next_readout) == absolute;
            batch.logits[local] = readout ? 1 : 0;
            if (readout) {
                wanted.push_back(static_cast<int32_t>(local));
                ++next_readout;
            }
        }
        if (llama_decode(context, batch) != 0) {
            throw std::runtime_error("Decode failed");
        }
        llama_synchronize(context);
        for (const int32_t local : wanted) {
            const float * logits = llama_get_logits_ith(context, local);
            if (!logits) throw std::runtime_error("Missing logits at a readout");
            json row = json::array();
            for (const int32_t token_id : label_ids) {
                row.push_back(static_cast<double>(logits[token_id]));
            }
            rows.push_back(std::move(row));
            std::vector<double> vocabulary_logits;
            vocabulary_logits.reserve(static_cast<size_t>(vocabulary_size));
            for (int token_id = 0; token_id < vocabulary_size; ++token_id) {
                vocabulary_logits.push_back(logits[token_id]);
            }
            const std::vector<int> allowed(label_ids.begin(), label_ids.end());
            const auto summary =
                riderless::summarize_vocabulary(vocabulary_logits, allowed);
            readouts.push_back({
                {"allowed_label_mass", summary.allowed_mass},
                {"full_vocabulary_argmax", {
                    {"token_id", summary.argmax_token_id},
                    {"logit", summary.argmax_logit},
                }},
            });
        }
        offset += count;
    }
    if (next_readout != sequence.positions.size()) {
        throw std::runtime_error("Readout accounting changed");
    }
    decode_ms = elapsed_ms(started);
    return rows;
}

json handle(
        llama_context * context,
        const llama_vocab * vocab,
        const settings & config,
        batch_holder & holder,
        const json & request) {
    const auto type = request.at("type").get<std::string>();
    if (type == "tokenize") {
        const auto text = request.at("text").get<std::string>();
        const bool add_special = request.value("add_special", false);
        return {{"ids", tokenize(vocab, text, add_special)}};
    }
    if (type != "scores") {
        throw std::runtime_error("Unknown request type: " + type);
    }
    const auto started = steady_clock::now();
    const auto label_ids = request.at("label_ids").get<std::vector<int32_t>>();
    if (label_ids.empty()) {
        throw std::runtime_error("label_ids must be nonempty");
    }
    const auto sequences = parse_sequences(request, config.context);
    json scores = json::array();
    json readouts = json::array();
    json prompt_tokens = json::array();
    json decode_times = json::array();
    for (const auto & sequence : sequences) {
        double decode_ms = 0.0;
        json sequence_readouts = json::array();
        scores.push_back(score_sequence(
            context, vocab, sequence, label_ids, config.batch, holder,
            sequence_readouts, decode_ms));
        readouts.push_back(std::move(sequence_readouts));
        prompt_tokens.push_back(sequence.ids.size());
        decode_times.push_back(decode_ms);
    }
    return {
        {"scores", std::move(scores)},
        {"readouts", std::move(readouts)},
        {"prompt_tokens", std::move(prompt_tokens)},
        {"decode_ms", std::move(decode_times)},
        {"total_ms", elapsed_ms(started)},
    };
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const settings config = parse_args(argc, argv);
        reject_incompatible_environment();
        ggml_backend_load_all_from_path(RIDERLESS_BACKEND_DIR);
        if (config.gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            throw std::runtime_error(
                "GPU opt-in requested but no GPU backend is available");
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
                sizeof(architecture)) < 1 ||
                std::string(architecture) != "gemma4") {
            throw std::runtime_error("Helper supports Gemma4 only");
        }
        char description[256];
        if (llama_model_desc(model.get(), description, sizeof(description)) < 1) {
            throw std::runtime_error("Model identity is unavailable");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());

        // Every field below is worker.cpp's sequential configuration
        // (riderless/api/native/worker.cpp), so the kernels selected here are
        // the kernels the reference run measured.
        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(config.context);
        context_params.n_batch = static_cast<uint32_t>(config.batch);
        context_params.n_ubatch = static_cast<uint32_t>(config.ubatch);
        context_params.n_seq_max = 1;
        context_params.kv_unified = false;
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
        batch_holder holder(static_cast<int32_t>(config.batch));

        const json hello = {
            {"type", "hello"},
            {"protocol_version", PROTOCOL_VERSION},
            {"model_name", std::string(description)},
            {"model_path", config.model_path},
            {"n_vocab", llama_vocab_n_tokens(vocab)},
            {"bos_token_id", llama_vocab_bos(vocab)},
            {"context_size", config.context},
            {"batch_size", config.batch},
            {"ubatch_size", config.ubatch},
            {"threads", config.threads},
            {"gpu", config.gpu},
            {"n_seq_max", 1},
            {"kv_unified", false},
            {"swa_full", true},
            {"attention", "causal"},
            {"flash_attn", "auto"},
            {"generated_tokens", 0},
        };
        std::cout << hello.dump() << '\n' << std::flush;

        std::string line;
        while (read_bounded_line(std::cin, line, MAX_PROTOCOL_BYTES)) {
            if (line.empty()) continue;
            std::string correlation_id = "unknown";
            json reply;
            try {
                const json request = json::parse(line);
                correlation_id = request.at("id").get<std::string>();
                reply = handle(context.get(), vocab, config, holder, request);
                reply["type"] = request.at("type");
            } catch (const std::exception & error) {
                reply = {{"type", "error"}, {"error", std::string(error.what())}};
            }
            reply["id"] = correlation_id;
            std::cout << reply.dump() << '\n' << std::flush;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
