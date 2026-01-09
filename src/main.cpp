#include <iostream>
#include <string>
#include <vector>

#include "compute/opencl_loader.h"
#include "sim/agent.h"
#include "sim/dna_memory.h"
#include "sim/environment.h"
#include "sim/fields.h"
#include "sim/io.h"
#include "sim/mycel.h"
#include "sim/params.h"
#include "sim/rng.h"

namespace {
struct CliOptions {
    bool width_set = false;
    bool height_set = false;
    SimParams params;
    uint32_t seed = 42;
    std::string resources_path;
    std::string pheromone_path;
    std::string molecules_path;
};

void print_help() {
    std::cout << "micro_swarm Optionen:\n"
              << "  --width N        Rasterbreite\n"
              << "  --height N       Rasterhoehe\n"
              << "  --agents N       Anzahl Agenten\n"
              << "  --steps N        Simulationsschritte\n"
              << "  --seed N         RNG-Seed\n"
              << "  --resources CSV  Startwerte Ressourcenfeld\n"
              << "  --pheromone CSV  Startwerte Pheromonfeld\n"
              << "  --molecules CSV  Startwerte Molekuelfeld\n"
              << "  --help           Hilfe anzeigen\n";
}

bool parse_int(const char *value, int &out) {
    try {
        out = std::stoi(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_seed(const char *value, uint32_t &out) {
    try {
        out = static_cast<uint32_t>(std::stoul(value));
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_cli(int argc, char **argv, CliOptions &opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return false;
        }
        if (i + 1 >= argc) {
            std::cerr << "Fehlender Wert fuer " << arg << "\n";
            return false;
        }
        const char *value = argv[++i];
        if (arg == "--width") {
            if (!parse_int(value, opts.params.width)) return false;
            opts.width_set = true;
        } else if (arg == "--height") {
            if (!parse_int(value, opts.params.height)) return false;
            opts.height_set = true;
        } else if (arg == "--agents") {
            if (!parse_int(value, opts.params.agent_count)) return false;
        } else if (arg == "--steps") {
            if (!parse_int(value, opts.params.steps)) return false;
        } else if (arg == "--seed") {
            if (!parse_seed(value, opts.seed)) return false;
        } else if (arg == "--resources") {
            opts.resources_path = value;
        } else if (arg == "--pheromone") {
            opts.pheromone_path = value;
        } else if (arg == "--molecules") {
            opts.molecules_path = value;
        } else {
            std::cerr << "Unbekanntes Argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}
} // namespace

int main(int argc, char **argv) {
    CliOptions opts;
    if (!parse_cli(argc, argv, opts)) {
        return 1;
    }
    SimParams params = opts.params;
    Rng rng(opts.seed);

    GridData resources_data;
    GridData pheromone_data;
    GridData molecules_data;
    std::string error;

    auto apply_dataset = [&](const std::string &path, GridData &data, const char *label) -> bool {
        if (path.empty()) return true;
        if (!load_grid_csv(path, data, error)) {
            std::cerr << label << ": " << error << "\n";
            return false;
        }
        if (opts.width_set && data.width != params.width) {
            std::cerr << "Breite aus CSV passt nicht zu --width\n";
            return false;
        }
        if (opts.height_set && data.height != params.height) {
            std::cerr << "Hoehe aus CSV passt nicht zu --height\n";
            return false;
        }
        params.width = data.width;
        params.height = data.height;
        return true;
    };

    if (!apply_dataset(opts.resources_path, resources_data, "resources")) return 1;
    if (!apply_dataset(opts.pheromone_path, pheromone_data, "pheromone")) return 1;
    if (!apply_dataset(opts.molecules_path, molecules_data, "molecules")) return 1;

    OpenCLStatus ocl = probe_opencl();
    std::cout << "[OpenCL] " << ocl.message << "\n";

    Environment env(params.width, params.height);
    if (!resources_data.values.empty()) {
        env.resources.data = resources_data.values;
    } else {
        env.seed_resources(rng);
    }

    GridField pheromone(params.width, params.height, 0.0f);
    GridField molecules(params.width, params.height, 0.0f);
    MycelNetwork mycel(params.width, params.height);
    if (!pheromone_data.values.empty()) {
        pheromone.data = pheromone_data.values;
    }
    if (!molecules_data.values.empty()) {
        molecules.data = molecules_data.values;
    }

    DNAMemory dna;
    std::vector<Agent> agents;
    agents.reserve(params.agent_count);

    for (int i = 0; i < params.agent_count; ++i) {
        Agent agent;
        agent.x = static_cast<float>(rng.uniform_int(0, params.width - 1));
        agent.y = static_cast<float>(rng.uniform_int(0, params.height - 1));
        agent.heading = rng.uniform(0.0f, 6.283185307f);
        agent.energy = rng.uniform(0.2f, 0.6f);
        agent.genome = dna.sample(rng, params);
        agents.push_back(agent);
    }

    FieldParams pheromone_params{params.pheromone_evaporation, params.pheromone_diffusion};
    FieldParams molecule_params{params.molecule_evaporation, params.molecule_diffusion};

    for (int step = 0; step < params.steps; ++step) {
        for (auto &agent : agents) {
            agent.step(rng, params, pheromone, molecules, env.resources);
            if (agent.energy > 1.2f) {
                dna.add(params, agent.genome, agent.energy);
                agent.energy *= 0.6f;
            }
        }

        diffuse_and_evaporate(pheromone, pheromone_params);
        diffuse_and_evaporate(molecules, molecule_params);

        mycel.update(params, pheromone, env.resources);
        env.regenerate(params);
        dna.decay();

        for (auto &agent : agents) {
            if (agent.energy <= 0.05f) {
                agent.x = static_cast<float>(rng.uniform_int(0, params.width - 1));
                agent.y = static_cast<float>(rng.uniform_int(0, params.height - 1));
                agent.heading = rng.uniform(0.0f, 6.283185307f);
                agent.energy = rng.uniform(0.2f, 0.5f);
                agent.genome = dna.sample(rng, params);
            }
        }

        if (step % 10 == 0) {
            float avg_energy = 0.0f;
            for (const auto &agent : agents) {
                avg_energy += agent.energy;
            }
            avg_energy /= static_cast<float>(agents.size());

            float mycel_sum = 0.0f;
            for (float v : mycel.density.data) {
                mycel_sum += v;
            }
            float mycel_avg = mycel_sum / static_cast<float>(mycel.density.data.size());

            std::cout << "step=" << step
                      << " avg_energy=" << avg_energy
                      << " dna_pool=" << dna.entries.size()
                      << " mycel_avg=" << mycel_avg
                      << "\n";
        }
    }

    std::cout << "done\n";
    return 0;
}
