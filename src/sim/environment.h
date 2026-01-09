#pragma once

#include "fields.h"
#include "params.h"
#include "rng.h"

struct Environment {
    GridField resources;
    int width = 0;
    int height = 0;

    Environment() = default;
    Environment(int w, int h);

    void seed_resources(Rng &rng);
    void regenerate(const SimParams &params);
};
