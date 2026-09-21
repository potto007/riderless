#include "worker-utils.h"

#include <cmath>
#include <stdexcept>
#include <vector>

static bool close(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

static void check(bool condition) {
    if (!condition) throw std::runtime_error("worker utility assertion failed");
}

int main() {
    {
        const auto probabilities = systemone::stable_softmax({0.0, std::log(3.0)});
        check(close(probabilities.at(0), 0.25));
        check(close(probabilities.at(1), 0.75));
    }
    {
        const auto probabilities = systemone::stable_softmax(
            {10000.0, 9999.0, -10000.0});
        check(close(probabilities.at(0) + probabilities.at(1) + probabilities.at(2), 1.0));
        check(probabilities.at(0) > probabilities.at(1));
        check(probabilities.at(2) >= 0.0);
    }
    {
        const std::vector<double> vocabulary{0.0, 1.0, 2.0, -1.0};
        const auto summary = systemone::summarize_vocabulary(vocabulary, {1, 3});
        const double denominator = 1.0 + std::exp(1.0) + std::exp(2.0) + std::exp(-1.0);
        check(close(summary.allowed_mass, (std::exp(1.0) + std::exp(-1.0)) / denominator));
        check(summary.argmax_token_id == 2);
        check(summary.argmax_logit == 2.0);
    }
    {
        check(systemone::single_suffix_token(
            std::vector<int>{1, 2}, std::vector<int>{1, 2, 7}) == 7);
        bool rejected = false;
        try {
            (void) systemone::single_suffix_token(
                std::vector<int>{1, 2}, std::vector<int>{1, 9, 7});
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected);
    }
    {
        const auto chunks = systemone::prefill_chunks(513, 256);
        check((chunks == std::vector<size_t>{256, 256, 1}));
        bool rejected = false;
        try {
            (void) systemone::prefill_chunks(0, 256);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected);
    }
    return 0;
}
