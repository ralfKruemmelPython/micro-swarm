#include "dna_memory.h"

#include <algorithm>

void DNAMemory::add(const SimParams &params, const Genome &genome, float fitness) {
    entries.push_back({genome, fitness, 0});
    std::sort(entries.begin(), entries.end(), [](const DNAEntry &a, const DNAEntry &b) {
        return a.fitness > b.fitness;
    });
    if (static_cast<int>(entries.size()) > params.dna_capacity) {
        entries.resize(params.dna_capacity);
    }
}

Genome DNAMemory::sample(Rng &rng, const SimParams &params) const {
    if (entries.empty()) {
        Genome g;
        g.sense_gain = rng.uniform(0.6f, 1.4f);
        g.pheromone_gain = rng.uniform(0.6f, 1.4f);
        g.exploration_bias = rng.uniform(0.2f, 0.8f);
        return g;
    }

    float total = 0.0f;
    for (const auto &entry : entries) {
        total += entry.fitness * params.dna_survival_bias + 0.01f;
    }

    float pick = rng.uniform(0.0f, total);
    for (const auto &entry : entries) {
        float w = entry.fitness * params.dna_survival_bias + 0.01f;
        if (pick <= w) {
            Genome g = entry.genome;
            g.sense_gain *= rng.uniform(0.9f, 1.1f);
            g.pheromone_gain *= rng.uniform(0.9f, 1.1f);
            g.exploration_bias = std::min(1.0f, std::max(0.0f, g.exploration_bias + rng.uniform(-0.05f, 0.05f)));
            return g;
        }
        pick -= w;
    }

    return entries.front().genome;
}

void DNAMemory::decay() {
    for (auto &entry : entries) {
        entry.age += 1;
        entry.fitness *= 0.995f;
    }
}
