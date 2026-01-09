#include "environment.h"

#include <algorithm>

Environment::Environment(int w, int h) : resources(w, h, 0.0f), blocked(static_cast<size_t>(w) * h, 0), width(w), height(h) {}

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
            if (!blocked.empty() && blocked[static_cast<size_t>(y) * width + x] != 0) {
                continue;
            }
            float &cell = resources.at(x, y);
            cell += params.resource_regen;
            if (cell > params.resource_max) {
                cell = params.resource_max;
            }
        }
    }
}

void Environment::apply_block_rect(int x, int y, int w, int h) {
    if (w <= 0 || h <= 0) {
        return;
    }
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(width, x + w);
    int y1 = std::min(height, y + h);
    for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
            resources.at(xx, yy) = 0.0f;
            if (!blocked.empty()) {
                blocked[static_cast<size_t>(yy) * width + xx] = 1;
            }
        }
    }
}

void Environment::shift_hotspots(int dx, int dy) {
    if (width <= 0 || height <= 0) {
        return;
    }
    std::vector<float> next(resources.data.size(), 0.0f);
    int sx = ((dx % width) + width) % width;
    int sy = ((dy % height) + height) % height;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int nx = (x + sx) % width;
            int ny = (y + sy) % height;
            next[static_cast<size_t>(ny) * width + nx] = resources.at(x, y);
        }
    }
    resources.data.swap(next);
}
