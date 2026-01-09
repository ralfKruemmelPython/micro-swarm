#pragma once

#include "fields.h"
#include "params.h"
#include "rng.h"

#include <cstdint>
#include <vector>

struct Environment {
    GridField resources;
    std::vector<uint8_t> blocked;
    int width = 0;
    int height = 0;

    Environment() = default;
    Environment(int w, int h);

    void seed_resources(Rng &rng);
    void regenerate(const SimParams &params);
    void apply_block_rect(int x, int y, int w, int h);
    void shift_hotspots(int dx, int dy);
};
