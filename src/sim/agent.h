#pragma once

#include "dna_memory.h"
#include "fields.h"
#include "params.h"
#include "rng.h"

struct SpeciesProfile {
    float exploration_mul = 1.0f;
    float food_attraction_mul = 1.0f;
    float danger_aversion_mul = 1.0f;
    float deposit_food_mul = 1.0f;
    float deposit_danger_mul = 1.0f;
};

struct Agent {
    float x = 0.0f;
    float y = 0.0f;
    float heading = 0.0f;
    float energy = 0.5f;
    float last_energy = 0.0f;
    float fitness_accum = 0.0f;
    int fitness_ticks = 0;
    float fitness_value = 0.0f;
    int species = 0;
    Genome genome;

    void step(Rng &rng,
              const SimParams &params,
              int fitness_window,
              const SpeciesProfile &profile,
              GridField &phero_food,
              GridField &phero_danger,
              GridField &molecules,
              GridField &resources);
};
