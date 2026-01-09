#include <iostream>
#include <vector>

#include "compute/opencl_loader.h"
#include "sim/agent.h"
#include "sim/dna_memory.h"
#include "sim/environment.h"
#include "sim/fields.h"
#include "sim/mycel.h"
#include "sim/params.h"
#include "sim/rng.h"

int main() {
    SimParams params;
    Rng rng(42);

    OpenCLStatus ocl = probe_opencl();
    std::cout << "[OpenCL] " << ocl.message << "\n";

    Environment env(params.width, params.height);
    env.seed_resources(rng);

    GridField pheromone(params.width, params.height, 0.0f);
    GridField molecules(params.width, params.height, 0.0f);
    MycelNetwork mycel(params.width, params.height);

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
