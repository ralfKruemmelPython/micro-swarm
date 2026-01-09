#include "agent.h"

#include <cmath>

namespace {
float wrap_angle(float a) {
    const float two_pi = 6.283185307f;
    while (a < 0.0f) a += two_pi;
    while (a >= two_pi) a -= two_pi;
    return a;
}

float sample_field(const GridField &field, float fx, float fy) {
    int x = static_cast<int>(fx);
    int y = static_cast<int>(fy);
    if (x < 0 || y < 0 || x >= field.width || y >= field.height) {
        return 0.0f;
    }
    return field.at(x, y);
}
} // namespace

void Agent::step(Rng &rng, const SimParams &params, GridField &pheromone, GridField &molecules, GridField &resources) {
    const float sensor = params.agent_sense_radius * genome.sense_gain;
    const float turn = params.agent_random_turn;

    float angles[3] = {
        heading - 0.6f,
        heading,
        heading + 0.6f
    };
    float weights[3] = {};

    for (int i = 0; i < 3; ++i) {
        float nx = x + std::cos(angles[i]) * sensor;
        float ny = y + std::sin(angles[i]) * sensor;
        float p = sample_field(pheromone, nx, ny) * genome.pheromone_gain;
        float r = sample_field(resources, nx, ny);
        float m = sample_field(molecules, nx, ny);
        weights[i] = p + r + 0.25f * m + 0.01f;
    }

    float total = weights[0] + weights[1] + weights[2];
    float pick = rng.uniform(0.0f, total);
    int choice = 1;
    for (int i = 0; i < 3; ++i) {
        if (pick <= weights[i]) {
            choice = i;
            break;
        }
        pick -= weights[i];
    }

    heading = wrap_angle(angles[choice] + rng.uniform(-turn, turn) * genome.exploration_bias);

    float nx = x + std::cos(heading);
    float ny = y + std::sin(heading);

    if (nx >= 0.0f && ny >= 0.0f && nx < pheromone.width && ny < pheromone.height) {
        x = nx;
        y = ny;
    } else {
        heading = wrap_angle(heading + 3.1415926f);
    }

    int cx = static_cast<int>(x);
    int cy = static_cast<int>(y);
    if (cx >= 0 && cy >= 0 && cx < resources.width && cy < resources.height) {
        float &cell = resources.at(cx, cy);
        float harvested = std::min(cell, params.agent_harvest);
        cell -= harvested;
        energy += harvested;

        float deposit = params.agent_deposit_scale * harvested;
        pheromone.at(cx, cy) += deposit;
        molecules.at(cx, cy) += harvested * 0.5f;
    }

    energy -= params.agent_move_cost;
    if (energy < 0.0f) {
        energy = 0.0f;
    }
}
