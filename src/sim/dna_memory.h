#pragma once

#include <vector>

#include "params.h"
#include "rng.h"

struct Genome {
    float sense_gain = 1.0f;
    float pheromone_gain = 1.0f;
    float exploration_bias = 0.5f;
};

struct DNAEntry {
    Genome genome;
    float fitness = 0.0f;
    int age = 0;
};

struct DNAMemory {
    std::vector<DNAEntry> entries;

    void add(const SimParams &params, const Genome &genome, float fitness);
    Genome sample(Rng &rng, const SimParams &params) const;
    void decay();
};
