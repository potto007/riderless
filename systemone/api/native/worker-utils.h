#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace systemone {

struct vocabulary_summary {
    double allowed_mass;
    int argmax_token_id;
    double argmax_logit;
};

inline std::vector<double> stable_softmax(const std::vector<double> & logits) {
    if (logits.size() < 2) throw std::runtime_error("Need at least two label logits");
    for (const double value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite label logit");
    }
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities;
    probabilities.reserve(logits.size());
    double total = 0.0;
    for (const double value : logits) {
        probabilities.push_back(std::exp(value - maximum));
        total += probabilities.back();
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::runtime_error("Invalid label normalization");
    }
    for (double & value : probabilities) value /= total;
    const double check = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (!std::isfinite(check) || std::abs(check - 1.0) > 1e-12) {
        throw std::runtime_error("Label probabilities are not normalized");
    }
    return probabilities;
}

inline vocabulary_summary summarize_vocabulary(
        const std::vector<double> & logits,
        const std::vector<int> & allowed_token_ids) {
    if (logits.empty()) throw std::runtime_error("Empty vocabulary logits");
    const auto argmax = std::max_element(logits.begin(), logits.end());
    for (const double value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite vocabulary logit");
    }
    const double maximum = *argmax;
    double denominator = 0.0;
    for (const double value : logits) denominator += std::exp(value - maximum);
    double numerator = 0.0;
    for (const int token_id : allowed_token_ids) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= logits.size()) {
            throw std::runtime_error("Allowed token is outside vocabulary");
        }
        numerator += std::exp(logits.at(static_cast<size_t>(token_id)) - maximum);
    }
    const double mass = numerator / denominator;
    if (!std::isfinite(mass) || mass < 0.0 || mass > 1.0) {
        throw std::runtime_error("Invalid allowed-label mass");
    }
    return {
        mass,
        static_cast<int>(std::distance(logits.begin(), argmax)),
        *argmax,
    };
}

template <typename Token>
Token single_suffix_token(
        const std::vector<Token> & prompt,
        const std::vector<Token> & extended) {
    if (extended.size() != prompt.size() + 1 ||
            !std::equal(prompt.begin(), prompt.end(), extended.begin())) {
        throw std::runtime_error("Label is not one suffix token for the exact prompt");
    }
    return extended.back();
}

inline std::vector<size_t> prefill_chunks(size_t token_count, size_t batch_size) {
    if (token_count == 0 || batch_size == 0) {
        throw std::runtime_error("Token and batch counts must be positive");
    }
    std::vector<size_t> chunks;
    for (size_t offset = 0; offset < token_count; offset += batch_size) {
        chunks.push_back(std::min(batch_size, token_count - offset));
    }
    return chunks;
}

}  // namespace systemone
