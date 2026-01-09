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

struct EvoParams {
    bool enabled = false;
    float elite_frac = 0.20f;
    float mutation_sigma = 0.05f;
    float exploration_delta = 0.05f;
    int fitness_window = 50;
    float age_decay = 0.995f;
};

struct DNAMemory {
    std::vector<DNAEntry> entries;

    void add(const SimParams &params, const Genome &genome, float fitness, const EvoParams &evo, int capacity_override = -1);
    Genome sample(Rng &rng, const SimParams &params, const EvoParams &evo) const;
    void decay(const EvoParams &evo);
};
