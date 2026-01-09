#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
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
#include "sim/report.h"
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
    int dump_every = 0;
    std::string dump_dir = "dumps";
    std::string dump_prefix = "swarm";
    std::string dump_subdir;
    std::string report_html_path;
    int report_downsample = 32;
    bool paper_mode = false;
    bool report_global_norm = false;
    int report_hist_bins = 64;
    bool report_include_sparklines = true;

    bool stress_enable = false;
    int stress_at_step = 120;
    bool stress_block_rect_set = false;
    int stress_block_x = 0;
    int stress_block_y = 0;
    int stress_block_w = 0;
    int stress_block_h = 0;
    bool stress_shift_set = false;
    int stress_shift_dx = 0;
    int stress_shift_dy = 0;
    float stress_pheromone_noise = 0.0f;
    uint32_t stress_seed = 0;
    bool stress_seed_set = false;

    bool evo_enable = false;
    float evo_elite_frac = 0.20f;
    float evo_min_energy_to_store = 1.6f;
    float evo_mutation_sigma = 0.05f;
    float evo_exploration_delta = 0.05f;
    int evo_fitness_window = 50;
    float evo_age_decay = 0.995f;
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
              << "  --mycel-growth F     Mycel-Wachstumsrate\n"
              << "  --mycel-decay F      Mycel-Decay\n"
              << "  --mycel-transport F  Mycel-Transport\n"
              << "  --mycel-threshold F  Mycel-Drive-Schwelle\n"
              << "  --mycel-drive-p F    Mycel-Drive-Gewicht Pheromon\n"
              << "  --mycel-drive-r F    Mycel-Drive-Gewicht Ressourcen\n"
              << "  --dump-every N   Dump-Intervall (0=aus)\n"
              << "  --dump-dir PATH  Dump-Verzeichnis\n"
              << "  --dump-prefix N  Dump-Dateiprefix\n"
              << "  [subdir]         Optionaler letzter Parameter: Unterordner in dump-dir\n"
              << "  --report-html PATH  Report-HTML-Pfad\n"
              << "  --report-downsample N  Report-Downsample (0=aus)\n"
              << "  --paper-mode           Paper-Modus aktivieren\n"
              << "  --report-global-norm   Globale Normalisierung fuer Previews\n"
              << "  --report-hist-bins N   Histogramm-Bins fuer Entropie\n"
              << "  --report-no-sparklines Sparklines deaktivieren\n"
              << "  --stress-enable                  Stress-Test aktivieren\n"
              << "  --stress-at-step N               Stress-Zeitpunkt\n"
              << "  --stress-block-rect x y w h      Ressourcen-Blockade\n"
              << "  --stress-shift-hotspots dx dy    Hotspots verschieben\n"
              << "  --stress-pheromone-noise F       Pheromon-Noise\n"
              << "  --stress-seed N                  Seed fuer Stress-Noise\n"
              << "  --evo-enable                     Evolution-Tuning aktivieren\n"
              << "  --evo-elite-frac F               Elite-Anteil\n"
              << "  --evo-min-energy-to-store F      Mindestenergie fuer Speicherung\n"
              << "  --evo-mutation-sigma F           Mutationsstaerke\n"
              << "  --evo-exploration-delta F        Exploration-Mutation\n"
              << "  --evo-fitness-window N           Fitness-Fenster\n"
              << "  --evo-age-decay F                Age-Decay pro Tick\n"
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

bool parse_float(const char *value, float &out) {
    try {
        out = std::stof(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_string(const char *value, std::string &out) {
    if (!value) {
        return false;
    }
    out = value;
    return !out.empty();
}

bool parse_cli(int argc, char **argv, CliOptions &opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return false;
        }
        if (!arg.empty() && arg[0] != '-' && i == argc - 1) {
            if (!parse_string(arg.c_str(), opts.dump_subdir)) {
                std::cerr << "Ungueltiger Wert fuer dump-subdir\n";
                return false;
            }
            continue;
        }
        if (arg == "--paper-mode") {
            opts.paper_mode = true;
            continue;
        }
        if (arg == "--report-global-norm") {
            opts.report_global_norm = true;
            continue;
        }
        if (arg == "--report-no-sparklines") {
            opts.report_include_sparklines = false;
            continue;
        }
        if (arg == "--stress-enable") {
            opts.stress_enable = true;
            continue;
        }
        if (arg == "--evo-enable") {
            opts.evo_enable = true;
            continue;
        }
        if (arg == "--stress-block-rect") {
            if (i + 4 >= argc) {
                std::cerr << "Fehlender Wert fuer " << arg << "\n";
                return false;
            }
            if (!parse_int(argv[i + 1], opts.stress_block_x) ||
                !parse_int(argv[i + 2], opts.stress_block_y) ||
                !parse_int(argv[i + 3], opts.stress_block_w) ||
                !parse_int(argv[i + 4], opts.stress_block_h)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.stress_block_rect_set = true;
            i += 4;
            continue;
        }
        if (arg == "--stress-shift-hotspots") {
            if (i + 2 >= argc) {
                std::cerr << "Fehlender Wert fuer " << arg << "\n";
                return false;
            }
            if (!parse_int(argv[i + 1], opts.stress_shift_dx) ||
                !parse_int(argv[i + 2], opts.stress_shift_dy)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.stress_shift_set = true;
            i += 2;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Fehlender Wert fuer " << arg << "\n";
            return false;
        }
        const char *value = argv[++i];
        if (arg == "--width") {
            if (!parse_int(value, opts.params.width)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.width_set = true;
        } else if (arg == "--height") {
            if (!parse_int(value, opts.params.height)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.height_set = true;
        } else if (arg == "--agents") {
            if (!parse_int(value, opts.params.agent_count)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--steps") {
            if (!parse_int(value, opts.params.steps)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--seed") {
            if (!parse_seed(value, opts.seed)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--resources") {
            opts.resources_path = value;
        } else if (arg == "--pheromone") {
            opts.pheromone_path = value;
        } else if (arg == "--molecules") {
            opts.molecules_path = value;
        } else if (arg == "--mycel-growth") {
            if (!parse_float(value, opts.params.mycel_growth)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--mycel-decay") {
            if (!parse_float(value, opts.params.mycel_decay)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--mycel-transport") {
            if (!parse_float(value, opts.params.mycel_transport)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--mycel-threshold") {
            if (!parse_float(value, opts.params.mycel_drive_threshold)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--mycel-drive-p") {
            if (!parse_float(value, opts.params.mycel_drive_p)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--mycel-drive-r") {
            if (!parse_float(value, opts.params.mycel_drive_r)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--dump-every") {
            if (!parse_int(value, opts.dump_every)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--dump-dir") {
            if (!parse_string(value, opts.dump_dir)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--dump-prefix") {
            if (!parse_string(value, opts.dump_prefix)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--report-html") {
            if (!parse_string(value, opts.report_html_path)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--report-downsample") {
            if (!parse_int(value, opts.report_downsample)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--report-hist-bins") {
            if (!parse_int(value, opts.report_hist_bins)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--stress-at-step") {
            if (!parse_int(value, opts.stress_at_step)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--stress-pheromone-noise") {
            if (!parse_float(value, opts.stress_pheromone_noise)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--stress-seed") {
            if (!parse_seed(value, opts.stress_seed)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.stress_seed_set = true;
        } else if (arg == "--evo-elite-frac") {
            if (!parse_float(value, opts.evo_elite_frac)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--evo-min-energy-to-store") {
            if (!parse_float(value, opts.evo_min_energy_to_store)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--evo-mutation-sigma") {
            if (!parse_float(value, opts.evo_mutation_sigma)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--evo-exploration-delta") {
            if (!parse_float(value, opts.evo_exploration_delta)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--evo-fitness-window") {
            if (!parse_int(value, opts.evo_fitness_window)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--evo-age-decay") {
            if (!parse_float(value, opts.evo_age_decay)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
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
    if (!opts.stress_seed_set) {
        opts.stress_seed = opts.seed;
    }

    if (opts.evo_enable) {
        if (opts.evo_elite_frac <= 0.0f || opts.evo_elite_frac > 1.0f) {
            std::cerr << "Ungueltiger Wert fuer --evo-elite-frac\n";
            return 1;
        }
        if (opts.evo_fitness_window <= 0) {
            std::cerr << "Ungueltiger Wert fuer --evo-fitness-window\n";
            return 1;
        }
        if (opts.evo_mutation_sigma < 0.0f || opts.evo_exploration_delta < 0.0f) {
            std::cerr << "Ungueltiger Wert fuer Evo-Mutationsparameter\n";
            return 1;
        }
        if (opts.evo_age_decay <= 0.0f || opts.evo_age_decay > 1.0f) {
            std::cerr << "Ungueltiger Wert fuer --evo-age-decay\n";
            return 1;
        }
    }
    if (opts.dump_every < 0) {
        std::cerr << "Ungueltiger Wert fuer --dump-every\n";
        return 1;
    }
    if (opts.report_downsample < 0) {
        std::cerr << "Ungueltiger Wert fuer --report-downsample\n";
        return 1;
    }
    if (opts.report_hist_bins <= 0) {
        std::cerr << "Ungueltiger Wert fuer --report-hist-bins\n";
        return 1;
    }
    if (!opts.dump_subdir.empty()) {
        std::filesystem::path base_dir = opts.dump_dir;
        std::filesystem::path sub_dir = opts.dump_subdir;
        opts.dump_dir = (base_dir / sub_dir).string();
        if (!opts.report_html_path.empty()) {
            std::filesystem::path report_path = opts.report_html_path;
            opts.report_html_path = (std::filesystem::path(opts.dump_dir) / report_path.filename()).string();
        }
    }

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
    EvoParams evo;
    evo.enabled = opts.evo_enable;
    evo.elite_frac = opts.evo_elite_frac;
    evo.mutation_sigma = opts.evo_mutation_sigma;
    evo.exploration_delta = opts.evo_exploration_delta;
    evo.fitness_window = opts.evo_fitness_window;
    evo.age_decay = opts.evo_age_decay;
    std::vector<Agent> agents;
    agents.reserve(params.agent_count);

    for (int i = 0; i < params.agent_count; ++i) {
        Agent agent;
        agent.x = static_cast<float>(rng.uniform_int(0, params.width - 1));
        agent.y = static_cast<float>(rng.uniform_int(0, params.height - 1));
        agent.heading = rng.uniform(0.0f, 6.283185307f);
        agent.energy = rng.uniform(0.2f, 0.6f);
        agent.genome = dna.sample(rng, params, evo);
        agents.push_back(agent);
    }

    FieldParams pheromone_params{params.pheromone_evaporation, params.pheromone_diffusion};
    FieldParams molecule_params{params.molecule_evaporation, params.molecule_diffusion};

    if (opts.dump_every > 0) {
        std::error_code ec;
        std::filesystem::create_directories(opts.dump_dir, ec);
        if (ec) {
            std::cerr << "Konnte Dump-Verzeichnis nicht erstellen: " << opts.dump_dir << "\n";
            return 1;
        }
    }

    auto dump_fields = [&](int step) -> bool {
        if (opts.dump_every <= 0) return true;
        if (step % opts.dump_every != 0) return true;

        std::ostringstream name;
        name << opts.dump_prefix << "_step" << std::setw(6) << std::setfill('0') << step;
        std::string base = name.str();

        std::string error;
        auto dump_one = [&](const std::string &suffix, const GridField &field) -> bool {
            std::filesystem::path path = std::filesystem::path(opts.dump_dir) / (base + suffix);
            if (!save_grid_csv(path.string(), field.width, field.height, field.data, error)) {
                std::cerr << error << "\n";
                return false;
            }
            return true;
        };

        if (!dump_one("_resources.csv", env.resources)) return false;
        if (!dump_one("_pheromone.csv", pheromone)) return false;
        if (!dump_one("_molecules.csv", molecules)) return false;
        if (!dump_one("_mycel.csv", mycel.density)) return false;
        return true;
    };

    bool stress_applied = false;
    Rng stress_rng(opts.stress_seed);
    std::vector<SystemMetrics> system_metrics;
    system_metrics.reserve(static_cast<size_t>(params.steps));

    for (int step = 0; step < params.steps; ++step) {
        if (opts.stress_enable && !stress_applied && step >= opts.stress_at_step) {
            if (opts.stress_block_rect_set) {
                env.apply_block_rect(opts.stress_block_x, opts.stress_block_y, opts.stress_block_w, opts.stress_block_h);
            }
            if (opts.stress_shift_set) {
                env.shift_hotspots(opts.stress_shift_dx, opts.stress_shift_dy);
            }
            stress_applied = true;
            std::cout << "[stress] applied at step=" << step << "\n";
        }
        if (!dump_fields(step)) {
            return 1;
        }
        for (auto &agent : agents) {
            agent.step(rng, params, opts.evo_enable ? opts.evo_fitness_window : 0, pheromone, molecules, env.resources);
            if (opts.evo_enable) {
                if (agent.energy > opts.evo_min_energy_to_store) {
                    dna.add(params, agent.genome, agent.fitness_value, evo);
                    agent.energy *= 0.6f;
                }
            } else {
                if (agent.energy > 1.2f) {
                    dna.add(params, agent.genome, agent.energy, evo);
                    agent.energy *= 0.6f;
                }
            }
        }

        diffuse_and_evaporate(pheromone, pheromone_params);
        diffuse_and_evaporate(molecules, molecule_params);

        if (opts.stress_enable && stress_applied && opts.stress_pheromone_noise > 0.0f) {
            for (float &v : pheromone.data) {
                v += stress_rng.uniform(0.0f, opts.stress_pheromone_noise);
                if (v < 0.0f) v = 0.0f;
            }
        }

        mycel.update(params, pheromone, env.resources);
        env.regenerate(params);
        dna.decay(evo);

        for (auto &agent : agents) {
            if (agent.energy <= 0.05f) {
                agent.x = static_cast<float>(rng.uniform_int(0, params.width - 1));
                agent.y = static_cast<float>(rng.uniform_int(0, params.height - 1));
                agent.heading = rng.uniform(0.0f, 6.283185307f);
                agent.energy = rng.uniform(0.2f, 0.5f);
                agent.last_energy = agent.energy;
                agent.fitness_accum = 0.0f;
                agent.fitness_ticks = 0;
                agent.fitness_value = 0.0f;
                agent.genome = dna.sample(rng, params, evo);
            }
        }

        float avg_energy = 0.0f;
        for (const auto &agent : agents) {
            avg_energy += agent.energy;
        }
        avg_energy /= static_cast<float>(agents.size());

        system_metrics.push_back({step, static_cast<int>(dna.entries.size()), avg_energy});

        if (step % 10 == 0) {
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

    if (opts.dump_every > 0) {
        ReportOptions report_opts;
        report_opts.dump_dir = opts.dump_dir;
        report_opts.dump_prefix = opts.dump_prefix;
        report_opts.report_html_path = opts.report_html_path;
        report_opts.downsample = opts.report_downsample;
        report_opts.paper_mode = opts.paper_mode;
        report_opts.global_normalization = opts.report_global_norm;
        report_opts.hist_bins = opts.report_hist_bins;
        report_opts.include_sparklines = opts.report_include_sparklines;
        report_opts.system_metrics = system_metrics;
        if (opts.stress_enable) {
            std::ostringstream scenario;
            scenario << "stress_enable=true";
            scenario << ", at_step=" << opts.stress_at_step;
            if (opts.stress_block_rect_set) {
                scenario << ", block_rect=" << opts.stress_block_x << "," << opts.stress_block_y << ","
                         << opts.stress_block_w << "," << opts.stress_block_h;
            }
            if (opts.stress_shift_set) {
                scenario << ", shift_hotspots=" << opts.stress_shift_dx << "," << opts.stress_shift_dy;
            }
            if (opts.stress_pheromone_noise > 0.0f) {
                scenario << ", pheromone_noise=" << opts.stress_pheromone_noise;
            }
            report_opts.scenario_summary = scenario.str();
        }
        std::string report_error;
        if (!generate_dump_report_html(report_opts, report_error)) {
            std::cerr << "Report-Fehler: " << report_error << "\n";
            return 1;
        }
        std::filesystem::path report_path;
        if (opts.report_html_path.empty()) {
            report_path = std::filesystem::path(opts.dump_dir) / (opts.dump_prefix + "_report.html");
        } else {
            report_path = opts.report_html_path;
        }
        std::cout << "report=" << report_path.string() << "\n";
    }

    std::cout << "done\n";
    return 0;
}
