#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <array>

#include "compute/opencl_loader.h"
#include "compute/opencl_runtime.h"
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

    bool ocl_enable = false;
    int ocl_device = 0;
    int ocl_platform = 0;
    bool ocl_print_devices = false;
    bool ocl_no_copyback = false;

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

    std::array<SpeciesProfile, 4> species_profiles;
    std::array<float, 4> species_fracs{0.40f, 0.25f, 0.20f, 0.15f};
    float global_spawn_frac = 0.15f;
};

std::array<SpeciesProfile, 4> default_species_profiles() {
    std::array<SpeciesProfile, 4> profiles;

    SpeciesProfile explorator;
    explorator.exploration_mul = 1.4f;
    explorator.food_attraction_mul = 0.6f;
    explorator.danger_aversion_mul = 0.8f;
    explorator.deposit_food_mul = 0.6f;
    explorator.deposit_danger_mul = 0.5f;
    explorator.resource_weight_mul = 1.4f;
    explorator.molecule_weight_mul = 1.4f;
    explorator.mycel_attraction_mul = 0.6f;
    explorator.novelty_weight = 0.6f;
    explorator.mutation_sigma_mul = 1.0f;
    explorator.exploration_delta_mul = 1.0f;
    explorator.dna_binding = 0.9f;

    SpeciesProfile integrator;
    integrator.exploration_mul = 0.7f;
    integrator.food_attraction_mul = 1.4f;
    integrator.danger_aversion_mul = 1.0f;
    integrator.deposit_food_mul = 1.5f;
    integrator.deposit_danger_mul = 0.8f;
    integrator.resource_weight_mul = 0.9f;
    integrator.molecule_weight_mul = 0.8f;
    integrator.mycel_attraction_mul = 1.5f;
    integrator.novelty_weight = 0.0f;
    integrator.mutation_sigma_mul = 1.0f;
    integrator.exploration_delta_mul = 1.0f;
    integrator.dna_binding = 1.0f;

    SpeciesProfile regulator;
    regulator.exploration_mul = 0.9f;
    regulator.food_attraction_mul = 0.8f;
    regulator.danger_aversion_mul = 1.8f;
    regulator.deposit_food_mul = 0.8f;
    regulator.deposit_danger_mul = 1.4f;
    regulator.resource_weight_mul = 0.9f;
    regulator.molecule_weight_mul = 0.8f;
    regulator.mycel_attraction_mul = 0.8f;
    regulator.novelty_weight = 0.0f;
    regulator.mutation_sigma_mul = 1.0f;
    regulator.exploration_delta_mul = 1.0f;
    regulator.dna_binding = 1.0f;
    regulator.over_density_threshold = 0.6f;
    regulator.counter_deposit_mul = 0.5f;

    SpeciesProfile innovator;
    innovator.exploration_mul = 1.3f;
    innovator.food_attraction_mul = 0.7f;
    innovator.danger_aversion_mul = 0.9f;
    innovator.deposit_food_mul = 0.7f;
    innovator.deposit_danger_mul = 0.7f;
    innovator.resource_weight_mul = 1.1f;
    innovator.molecule_weight_mul = 1.2f;
    innovator.mycel_attraction_mul = 0.6f;
    innovator.novelty_weight = 0.8f;
    innovator.mutation_sigma_mul = 1.6f;
    innovator.exploration_delta_mul = 1.6f;
    innovator.dna_binding = 0.6f;

    profiles[0] = explorator;
    profiles[1] = integrator;
    profiles[2] = regulator;
    profiles[3] = innovator;
    return profiles;
}

int pick_species(Rng &rng, const std::array<float, 4> &fracs) {
    float r = rng.uniform(0.0f, 1.0f);
    float accum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        accum += fracs[i];
        if (r <= accum) {
            return i;
        }
    }
    return 3;
}

void print_help() {
    std::cout << "micro_swarm Optionen:\n"
              << "  --width N        Rasterbreite\n"
              << "  --height N       Rasterhoehe\n"
              << "  --size N         Setzt Breite und Hoehe gleich\n"
              << "  --agents N       Anzahl Agenten\n"
              << "  --steps N        Simulationsschritte\n"
              << "  --seed N         RNG-Seed\n"
              << "  --resources CSV  Startwerte Ressourcenfeld\n"
              << "  --pheromone CSV  Startwerte Pheromonfeld\n"
              << "  --molecules CSV  Startwerte Molekuelfeld\n"
              << "  --resource-regen F  Ressourcen-Regeneration\n"
              << "  --mycel-growth F     Mycel-Wachstumsrate\n"
              << "  --mycel-decay F      Mycel-Decay\n"
              << "  --mycel-transport F  Mycel-Transport\n"
              << "  --mycel-threshold F  Mycel-Drive-Schwelle\n"
              << "  --mycel-drive-p F    Mycel-Drive-Gewicht Pheromon\n"
              << "  --mycel-drive-r F    Mycel-Drive-Gewicht Ressourcen\n"
              << "  --phero-food-deposit F     Pheromon Food Deposit\n"
              << "  --phero-danger-deposit F   Pheromon Danger Deposit\n"
              << "  --danger-delta-threshold F Danger Delta Schwelle\n"
              << "  --danger-bounce-deposit F  Danger Deposit bei Bounce\n"
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
              << "  --ocl-enable           OpenCL Diffusion aktivieren\n"
              << "  --ocl-device N         OpenCL Device Index\n"
              << "  --ocl-platform N       OpenCL Platform Index\n"
              << "  --ocl-print-devices    OpenCL Platforms/Devices auflisten\n"
              << "  --ocl-no-copyback      Host-Backcopy nur bei Dump/Ende\n"
              << "  --gpu N                Alias fuer OpenCL (0=aus, 1=an)\n"
              << "  --species-fracs f0 f1 f2 f3           Spezies-Anteile\n"
              << "  --species-profile S e f d df dd       Spezies-Profilwerte\n"
              << "  --global-spawn-frac F                 Anteil Spawn aus Global-Pool\n"
              << "  --dna-global-capacity N               Kapazitaet Global-Pool\n"
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
        if (arg == "--ocl-enable") {
            opts.ocl_enable = true;
            continue;
        }
        if (arg == "--ocl-print-devices") {
            opts.ocl_print_devices = true;
            continue;
        }
        if (arg == "--ocl-no-copyback") {
            opts.ocl_no_copyback = true;
            continue;
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
        if (arg == "--species-fracs") {
            if (i + 4 >= argc) {
                std::cerr << "Fehlender Wert fuer " << arg << "\n";
                return false;
            }
            for (int s = 0; s < 4; ++s) {
                float v = 0.0f;
                if (!parse_float(argv[i + 1 + s], v)) {
                    std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                    return false;
                }
                opts.species_fracs[s] = v;
            }
            i += 4;
            continue;
        }
        if (arg == "--species-profile") {
            if (i + 6 >= argc) {
                std::cerr << "Fehlender Wert fuer " << arg << "\n";
                return false;
            }
            int s = 0;
            if (!parse_int(argv[i + 1], s) || s < 0 || s > 3) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            float e = 0.0f, fa = 0.0f, da = 0.0f, df = 0.0f, dd = 0.0f;
            if (!parse_float(argv[i + 2], e) ||
                !parse_float(argv[i + 3], fa) ||
                !parse_float(argv[i + 4], da) ||
                !parse_float(argv[i + 5], df) ||
                !parse_float(argv[i + 6], dd)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.species_profiles[s].exploration_mul = e;
            opts.species_profiles[s].food_attraction_mul = fa;
            opts.species_profiles[s].danger_aversion_mul = da;
            opts.species_profiles[s].deposit_food_mul = df;
            opts.species_profiles[s].deposit_danger_mul = dd;
            i += 6;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Fehlender Wert fuer " << arg << "\n";
            return false;
        }
        const char *value = argv[++i];
        if (arg == "--width" || arg == "--wight") {
            if (!parse_int(value, opts.params.width)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.width_set = true;
        } else if (arg == "--height" || arg == "--hight") {
            if (!parse_int(value, opts.params.height)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.height_set = true;
        } else if (arg == "--size") {
            int size = 0;
            if (!parse_int(value, size)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.params.width = size;
            opts.params.height = size;
            opts.width_set = true;
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
        } else if (arg == "--resource-regen") {
            if (!parse_float(value, opts.params.resource_regen)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
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
        } else if (arg == "--phero-food-deposit") {
            if (!parse_float(value, opts.params.phero_food_deposit_scale)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--phero-danger-deposit") {
            if (!parse_float(value, opts.params.phero_danger_deposit_scale)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--danger-delta-threshold") {
            if (!parse_float(value, opts.params.danger_delta_threshold)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--danger-bounce-deposit") {
            if (!parse_float(value, opts.params.danger_bounce_deposit)) {
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
        } else if (arg == "--global-spawn-frac") {
            if (!parse_float(value, opts.global_spawn_frac)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--dna-global-capacity") {
            if (!parse_int(value, opts.params.dna_global_capacity)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--gpu") {
            int gpu = 0;
            if (!parse_int(value, gpu) || (gpu != 0 && gpu != 1)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
            opts.ocl_enable = (gpu == 1);
        } else if (arg == "--ocl-device") {
            if (!parse_int(value, opts.ocl_device)) {
                std::cerr << "Ungueltiger Wert fuer " << arg << "\n";
                return false;
            }
        } else if (arg == "--ocl-platform") {
            if (!parse_int(value, opts.ocl_platform)) {
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
    opts.species_profiles = default_species_profiles();
    if (!parse_cli(argc, argv, opts)) {
        return 1;
    }
    if (opts.ocl_print_devices) {
        std::string output;
        std::string err;
        if (!OpenCLRuntime::print_devices(output, err)) {
            std::cerr << "[OpenCL] " << err << "\n";
            return 1;
        }
        std::cout << output;
        return 0;
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
    if (opts.global_spawn_frac < 0.0f || opts.global_spawn_frac > 1.0f) {
        std::cerr << "Ungueltiger Wert fuer --global-spawn-frac\n";
        return 1;
    }
    if (params.dna_global_capacity <= 0) {
        std::cerr << "Ungueltiger Wert fuer --dna-global-capacity\n";
        return 1;
    }
    {
        float sum = 0.0f;
        for (float f : opts.species_fracs) {
            if (f < 0.0f) {
                std::cerr << "Ungueltiger Wert fuer --species-fracs\n";
                return 1;
            }
            sum += f;
        }
        if (std::abs(sum - 1.0f) > 1e-3f) {
            std::cerr << "Ungueltige Summe fuer --species-fracs (muss ~1.0 sein)\n";
            return 1;
        }
    }
    if (opts.ocl_no_copyback && params.agent_count > 0) {
        std::cerr << "[OpenCL] ocl-no-copyback ist mit aktiven Agenten nicht kompatibel, erzwungenes Copyback.\n";
        opts.ocl_no_copyback = false;
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

    OpenCLStatus ocl_probe = probe_opencl();
    std::cout << "[OpenCL] " << ocl_probe.message << "\n";

    Environment env(params.width, params.height);
    if (!resources_data.values.empty()) {
        env.resources.data = resources_data.values;
    } else {
        env.seed_resources(rng);
    }

    GridField phero_food(params.width, params.height, 0.0f);
    GridField phero_danger(params.width, params.height, 0.0f);
    GridField molecules(params.width, params.height, 0.0f);
    MycelNetwork mycel(params.width, params.height);
    if (!pheromone_data.values.empty()) {
        phero_food.data = pheromone_data.values;
    }
    if (!molecules_data.values.empty()) {
        molecules.data = molecules_data.values;
    }

    std::array<DNAMemory, 4> dna_species;
    DNAMemory dna_global;
    EvoParams evo;
    evo.enabled = opts.evo_enable;
    evo.elite_frac = opts.evo_elite_frac;
    evo.mutation_sigma = opts.evo_mutation_sigma;
    evo.exploration_delta = opts.evo_exploration_delta;
    evo.fitness_window = opts.evo_fitness_window;
    evo.age_decay = opts.evo_age_decay;
    std::vector<Agent> agents;
    agents.reserve(params.agent_count);

    auto random_genome = [&]() -> Genome {
        Genome g;
        g.sense_gain = rng.uniform(0.6f, 1.4f);
        g.pheromone_gain = rng.uniform(0.6f, 1.4f);
        g.exploration_bias = rng.uniform(0.2f, 0.8f);
        return g;
    };

    auto apply_role_mutation = [&](Genome &g, const SpeciesProfile &profile) {
        float sigma = evo.mutation_sigma * profile.mutation_sigma_mul;
        float delta = evo.exploration_delta * profile.exploration_delta_mul;
        if (sigma > 0.0f) {
            g.sense_gain *= rng.uniform(1.0f - sigma, 1.0f + sigma);
            g.pheromone_gain *= rng.uniform(1.0f - sigma, 1.0f + sigma);
        }
        if (delta > 0.0f) {
            g.exploration_bias += rng.uniform(-delta, delta);
        }
        g.sense_gain = std::min(3.0f, std::max(0.2f, g.sense_gain));
        g.pheromone_gain = std::min(3.0f, std::max(0.2f, g.pheromone_gain));
        g.exploration_bias = std::min(1.0f, std::max(0.0f, g.exploration_bias));
    };

    auto sample_genome = [&](int species) -> Genome {
        const SpeciesProfile &profile = opts.species_profiles[species];
        bool use_dna = rng.uniform(0.0f, 1.0f) < profile.dna_binding;
        Genome g;
        if (use_dna) {
            if (opts.evo_enable && !dna_global.entries.empty() && rng.uniform(0.0f, 1.0f) < opts.global_spawn_frac) {
                g = dna_global.sample(rng, params, evo);
            } else {
                g = dna_species[species].sample(rng, params, evo);
            }
        } else {
            g = random_genome();
        }
        if (opts.evo_enable) {
            apply_role_mutation(g, profile);
        }
        return g;
    };

    const float global_epsilon = 1e-6f;
    auto maybe_add_global = [&](const Genome &genome, float fitness) {
        if (!opts.evo_enable) {
            return;
        }
        if (params.dna_global_capacity <= 0) {
            return;
        }
        if (dna_global.entries.size() < static_cast<size_t>(params.dna_global_capacity)) {
            dna_global.add(params, genome, fitness, evo, params.dna_global_capacity);
            return;
        }
        float worst = dna_global.entries.back().fitness;
        if (fitness > worst + global_epsilon) {
            dna_global.add(params, genome, fitness, evo, params.dna_global_capacity);
        }
    };

    for (int i = 0; i < params.agent_count; ++i) {
        Agent agent;
        agent.x = static_cast<float>(rng.uniform_int(0, params.width - 1));
        agent.y = static_cast<float>(rng.uniform_int(0, params.height - 1));
        agent.heading = rng.uniform(0.0f, 6.283185307f);
        agent.energy = rng.uniform(0.2f, 0.6f);
        agent.species = pick_species(rng, opts.species_fracs);
        agent.genome = sample_genome(agent.species);
        agents.push_back(agent);
    }

    FieldParams pheromone_params{params.pheromone_evaporation, params.pheromone_diffusion};
    FieldParams molecule_params{params.molecule_evaporation, params.molecule_diffusion};

    OpenCLRuntime ocl_runtime;
    bool ocl_active = false;
    if (opts.ocl_enable) {
        std::string ocl_error;
        if (!ocl_runtime.init(opts.ocl_platform, opts.ocl_device, ocl_error)) {
            std::cerr << "[OpenCL] init failed, fallback to CPU: " << ocl_error << "\n";
        } else if (!ocl_runtime.build_kernels(ocl_error)) {
            std::cerr << "[OpenCL] kernel build failed, fallback to CPU: " << ocl_error << "\n";
        } else if (!ocl_runtime.init_fields(phero_food, phero_danger, molecules, ocl_error)) {
            std::cerr << "[OpenCL] buffer init failed, fallback to CPU: " << ocl_error << "\n";
        } else {
            std::cout << "[OpenCL] platform/device: " << ocl_runtime.device_info() << "\n";
            std::cout << "[OpenCL] kernels built\n";
            ocl_active = true;
        }
    }

    auto run_ocl_self_test = [&](OpenCLRuntime &runtime) -> bool {
        GridField pf(16, 16, 0.0f);
        GridField pd(16, 16, 0.0f);
        GridField m(16, 16, 0.0f);
        for (int y = 0; y < pf.height; ++y) {
            for (int x = 0; x < pf.width; ++x) {
                float v = rng.uniform(0.0f, 1.0f);
                pf.at(x, y) = v;
                pd.at(x, y) = 1.0f - v;
                m.at(x, y) = 1.0f - v;
            }
        }
        GridField cpu_pf = pf;
        GridField cpu_pd = pd;
        GridField cpu_m = m;
        FieldParams fp{0.02f, 0.15f};
        FieldParams fm{0.35f, 0.25f};
        for (int i = 0; i < 5; ++i) {
            diffuse_and_evaporate(cpu_pf, fp);
            diffuse_and_evaporate(cpu_pd, fp);
            diffuse_and_evaporate(cpu_m, fm);
        }

        std::string error;
        if (!runtime.init_fields(pf, pd, m, error)) {
            std::cerr << "[OpenCL] self-test init failed: " << error << "\n";
            return false;
        }
        for (int i = 0; i < 5; ++i) {
            if (!runtime.step_diffuse(fp, fm, true, pf, pd, m, error)) {
                std::cerr << "[OpenCL] self-test step failed: " << error << "\n";
                return false;
            }
        }
        double mean_diff = 0.0;
        double max_abs = 0.0;
        for (size_t i = 0; i < pf.data.size(); ++i) {
            double d1 = std::abs(static_cast<double>(pf.data[i]) - cpu_pf.data[i]);
            double d2 = std::abs(static_cast<double>(pd.data[i]) - cpu_pd.data[i]);
            mean_diff += d1 + d2;
            if (d1 > max_abs) max_abs = d1;
            if (d2 > max_abs) max_abs = d2;
        }
        mean_diff /= static_cast<double>(pf.data.size() * 2);
        std::cout << "[OpenCL] self-test mean_diff=" << mean_diff << " max_abs=" << max_abs << "\n";
        if (max_abs > 1e-3) {
            std::cerr << "[OpenCL] self-test too large diff, fallback to CPU\n";
            return false;
        }
        return true;
    };

    if (ocl_active) {
        if (!run_ocl_self_test(ocl_runtime)) {
            ocl_active = false;
        } else {
            std::string ocl_error;
            if (!ocl_runtime.init_fields(phero_food, phero_danger, molecules, ocl_error)) {
                std::cerr << "[OpenCL] buffer init failed, fallback to CPU: " << ocl_error << "\n";
                ocl_active = false;
            } else {
                std::cout << "[OpenCL] using GPU diffusion\n";
                if (opts.ocl_no_copyback) {
                    std::cout << "[OpenCL] no-copyback enabled\n";
                }
            }
        }
    }

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
        if (!dump_one("_phero_food.csv", phero_food)) return false;
        if (!dump_one("_phero_danger.csv", phero_danger)) return false;
        if (!dump_one("_molecules.csv", molecules)) return false;
        if (!dump_one("_mycel.csv", mycel.density)) return false;
        return true;
    };

    bool stress_applied = false;
    Rng stress_rng(opts.stress_seed);
    std::vector<SystemMetrics> system_metrics;
    system_metrics.reserve(static_cast<size_t>(params.steps));

    for (int step = 0; step < params.steps; ++step) {
        bool dump_step = (opts.dump_every > 0 && step % opts.dump_every == 0);
        if (ocl_active && opts.ocl_no_copyback && dump_step) {
            std::string ocl_error;
            if (!ocl_runtime.copyback(phero_food, phero_danger, molecules, ocl_error)) {
                std::cerr << "[OpenCL] copyback failed, fallback to CPU: " << ocl_error << "\n";
                ocl_active = false;
            }
        }
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
            const SpeciesProfile &profile = opts.species_profiles[agent.species];
            agent.step(rng, params, opts.evo_enable ? opts.evo_fitness_window : 0, profile, phero_food, phero_danger, molecules, env.resources, mycel.density);
            if (opts.evo_enable) {
                if (agent.energy > opts.evo_min_energy_to_store) {
                    dna_species[agent.species].add(params, agent.genome, agent.fitness_value, evo, params.dna_capacity);
                    maybe_add_global(agent.genome, agent.fitness_value);
                    agent.energy *= 0.6f;
                }
            } else {
                if (agent.energy > 1.2f) {
                    dna_species[agent.species].add(params, agent.genome, agent.energy, evo, params.dna_capacity);
                    agent.energy *= 0.6f;
                }
            }
        }

        if (ocl_active) {
            std::string ocl_error;
            if (!ocl_runtime.upload_fields(phero_food, phero_danger, molecules, ocl_error)) {
                std::cerr << "[OpenCL] upload failed, fallback to CPU: " << ocl_error << "\n";
                ocl_active = false;
            }
        }

        if (ocl_active) {
            bool do_copyback = (!opts.ocl_no_copyback) || dump_step;
            std::string ocl_error;
            if (!ocl_runtime.step_diffuse(pheromone_params, molecule_params, do_copyback, phero_food, phero_danger, molecules, ocl_error)) {
                std::cerr << "[OpenCL] diffuse failed, fallback to CPU: " << ocl_error << "\n";
                ocl_active = false;
                diffuse_and_evaporate(phero_food, pheromone_params);
                diffuse_and_evaporate(phero_danger, pheromone_params);
                diffuse_and_evaporate(molecules, molecule_params);
            }
        } else {
            diffuse_and_evaporate(phero_food, pheromone_params);
            diffuse_and_evaporate(phero_danger, pheromone_params);
            diffuse_and_evaporate(molecules, molecule_params);
        }

        if (opts.stress_enable && stress_applied && opts.stress_pheromone_noise > 0.0f) {
            for (float &v : phero_food.data) {
                v += stress_rng.uniform(0.0f, opts.stress_pheromone_noise);
                if (v < 0.0f) v = 0.0f;
            }
            for (float &v : phero_danger.data) {
                v += stress_rng.uniform(0.0f, opts.stress_pheromone_noise);
                if (v < 0.0f) v = 0.0f;
            }
        }

        mycel.update(params, phero_food, env.resources);
        env.regenerate(params);
        for (auto &pool : dna_species) {
            pool.decay(evo);
        }
        dna_global.decay(evo);

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
                agent.species = pick_species(rng, opts.species_fracs);
                agent.genome = sample_genome(agent.species);
            }
        }

        float avg_energy = 0.0f;
        std::array<float, 4> energy_sum{0.0f, 0.0f, 0.0f, 0.0f};
        std::array<int, 4> energy_count{0, 0, 0, 0};
        for (const auto &agent : agents) {
            avg_energy += agent.energy;
            if (agent.species >= 0 && agent.species < 4) {
                energy_sum[agent.species] += agent.energy;
                energy_count[agent.species] += 1;
            }
        }
        avg_energy /= static_cast<float>(agents.size());

        SystemMetrics m;
        m.step = step;
        m.avg_agent_energy = avg_energy;
        int dna_total = 0;
        for (int s = 0; s < 4; ++s) {
            m.dna_species_sizes[s] = static_cast<int>(dna_species[s].entries.size());
            dna_total += m.dna_species_sizes[s];
            if (energy_count[s] > 0) {
                m.avg_energy_by_species[s] = energy_sum[s] / static_cast<float>(energy_count[s]);
            } else {
                m.avg_energy_by_species[s] = 0.0f;
            }
        }
        m.dna_global_size = static_cast<int>(dna_global.entries.size());
        m.dna_pool_size = dna_total;
        system_metrics.push_back(m);

        if (step % 10 == 0) {
            float mycel_sum = 0.0f;
            for (float v : mycel.density.data) {
                mycel_sum += v;
            }
            float mycel_avg = mycel_sum / static_cast<float>(mycel.density.data.size());

            std::cout << "step=" << step
                      << " avg_energy=" << avg_energy
                      << " dna_pool=" << dna_total
                      << " mycel_avg=" << mycel_avg
                      << "\n";
        }
    }

    if (ocl_active && opts.ocl_no_copyback) {
        std::string ocl_error;
        if (!ocl_runtime.copyback(phero_food, phero_danger, molecules, ocl_error)) {
            std::cerr << "[OpenCL] final copyback failed: " << ocl_error << "\n";
            return 1;
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
