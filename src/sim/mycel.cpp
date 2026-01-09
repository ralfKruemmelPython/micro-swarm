#include "mycel.h"

#include <algorithm>

MycelNetwork::MycelNetwork(int w, int h) : density(w, h, 0.0f), width(w), height(h) {}

void MycelNetwork::update(const SimParams &params, const GridField &pheromone, const GridField &resources) {
    std::vector<float> next(density.data.size(), 0.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float current = density.at(x, y);
            float local_pheromone = pheromone.at(x, y);
            float local_resource = resources.at(x, y);

            float growth = params.mycel_growth * (local_pheromone + local_resource);
            float decay = params.mycel_decay * current;

            float neighbor_support = 0.0f;
            auto add = [&](int nx, int ny) {
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                    return;
                }
                neighbor_support += density.at(nx, ny);
            };

            add(x - 1, y);
            add(x + 1, y);
            add(x, y - 1);
            add(x, y + 1);

            float transport = params.mycel_transport * neighbor_support * 0.25f;
            float value = current + growth + transport - decay;
            next[y * width + x] = std::max(0.0f, std::min(1.0f, value));
        }
    }

    density.data.swap(next);
}
