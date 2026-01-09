#pragma once

#include <random>

struct Rng {
    std::mt19937 rng;

    explicit Rng(uint32_t seed) : rng(seed) {}

    float uniform(float a = 0.0f, float b = 1.0f) {
        std::uniform_real_distribution<float> dist(a, b);
        return dist(rng);
    }

    int uniform_int(int a, int b) {
        std::uniform_int_distribution<int> dist(a, b);
        return dist(rng);
    }
};
