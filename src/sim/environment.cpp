#include "environment.h"

Environment::Environment(int w, int h) : resources(w, h, 0.0f), width(w), height(h) {}

void Environment::seed_resources(Rng &rng) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float v = rng.uniform(0.0f, 1.0f);
            resources.at(x, y) = (v > 0.98f) ? rng.uniform(0.5f, 1.0f) : 0.0f;
        }
    }
}

void Environment::regenerate(const SimParams &params) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float &cell = resources.at(x, y);
            cell += params.resource_regen;
            if (cell > params.resource_max) {
                cell = params.resource_max;
            }
        }
    }
}
