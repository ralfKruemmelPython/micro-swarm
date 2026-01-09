#pragma once

#include "dna_memory.h"
#include "fields.h"
#include "params.h"
#include "rng.h"

struct Agent {
    float x = 0.0f;
    float y = 0.0f;
    float heading = 0.0f;
    float energy = 0.5f;
    float last_energy = 0.0f;
    float fitness_accum = 0.0f;
    int fitness_ticks = 0;
    float fitness_value = 0.0f;
    Genome genome;

    void step(Rng &rng, const SimParams &params, int fitness_window, GridField &pheromone, GridField &molecules, GridField &resources);
};
