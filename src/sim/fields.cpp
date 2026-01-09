#include "fields.h"

#include <algorithm>

GridField::GridField(int w, int h, float value) : width(w), height(h), data(w * h, value) {}

float &GridField::at(int x, int y) {
    return data[y * width + x];
}

float GridField::at(int x, int y) const {
    return data[y * width + x];
}

void GridField::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void diffuse_and_evaporate(GridField &field, const FieldParams &params) {
    std::vector<float> next(field.data.size(), 0.0f);
    const float diff = params.diffusion;
    const float evap = params.evaporation;

    for (int y = 0; y < field.height; ++y) {
        for (int x = 0; x < field.width; ++x) {
            float center = field.at(x, y);
            float sum = center * (1.0f - diff);
            int count = 0;

            auto add = [&](int nx, int ny) {
                if (nx < 0 || ny < 0 || nx >= field.width || ny >= field.height) {
                    return;
                }
                sum += field.at(nx, ny) * (diff * 0.25f);
                count++;
            };

            add(x - 1, y);
            add(x + 1, y);
            add(x, y - 1);
            add(x, y + 1);

            float value = sum;
            if (count < 4) {
                value = center;
            }
            value *= (1.0f - evap);
            next[y * field.width + x] = std::max(0.0f, value);
        }
    }

    field.data.swap(next);
}
