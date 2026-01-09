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
    Genome genome;

    void step(Rng &rng, const SimParams &params, GridField &pheromone, GridField &molecules, GridField &resources);
};
