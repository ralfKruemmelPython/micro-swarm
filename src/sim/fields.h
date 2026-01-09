#pragma once

#include <vector>

struct GridField {
    int width = 0;
    int height = 0;
    std::vector<float> data;

    GridField() = default;
    GridField(int w, int h, float value = 0.0f);

    float &at(int x, int y);
    float at(int x, int y) const;

    void fill(float value);
};

struct FieldParams {
    float evaporation = 0.0f;
    float diffusion = 0.0f;
};

void diffuse_and_evaporate(GridField &field, const FieldParams &params);
