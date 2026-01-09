#pragma once

#include <cstdint>

struct SimParams {
    int width = 128;
    int height = 128;
    int agent_count = 512;
    int steps = 200;

    float pheromone_evaporation = 0.02f;
    float pheromone_diffusion = 0.15f;
    float molecule_evaporation = 0.35f;
    float molecule_diffusion = 0.25f;

    float resource_regen = 0.0015f;
    float resource_max = 1.0f;

    float mycel_decay = 0.003f;
    float mycel_growth = 0.02f;
    float mycel_transport = 0.12f;
    float mycel_drive_threshold = 0.08f;
    float mycel_drive_p = 0.6f;
    float mycel_drive_r = 0.4f;

    float agent_move_cost = 0.01f;
    float agent_harvest = 0.04f;
    float agent_deposit_scale = 0.8f;
    float agent_sense_radius = 2.5f;
    float agent_random_turn = 0.2f;

    int dna_capacity = 256;
    int dna_global_capacity = 128;
    float dna_survival_bias = 0.7f;

    float phero_food_deposit_scale = 0.8f;
    float phero_danger_deposit_scale = 0.6f;
    float danger_delta_threshold = 0.05f;
    float danger_bounce_deposit = 0.02f;
};
