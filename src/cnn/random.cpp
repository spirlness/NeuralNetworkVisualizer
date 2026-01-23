#include "cnn/random.h"
#include <random>

namespace {
    std::mt19937 createRng() {
        std::random_device rd;
        return std::mt19937(rd());
    }
}

std::mt19937& getRng() {
    static std::mt19937 rng = createRng();
    return rng;
}
