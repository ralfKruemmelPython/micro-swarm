#pragma once

#include <string>
#include <vector>

#include "sim/fields.h"

class OpenCLRuntime {
public:
    OpenCLRuntime();
    ~OpenCLRuntime();

    bool init(int platform_index, int device_index, std::string &error);
    bool build_kernels(std::string &error);
    bool init_fields(const GridField &phero_food,
                     const GridField &phero_danger,
                     const GridField &molecules,
                     std::string &error);
    bool upload_fields(const GridField &phero_food,
                       const GridField &phero_danger,
                       const GridField &molecules,
                       std::string &error);
    bool step_diffuse(const FieldParams &pheromone_params,
                      const FieldParams &molecule_params,
                      bool do_copyback,
                      GridField &phero_food,
                      GridField &phero_danger,
                      GridField &molecules,
                      std::string &error);
    bool copyback(GridField &phero_food, GridField &phero_danger, GridField &molecules, std::string &error);
    bool is_available() const;
    std::string device_info() const;

    static bool print_devices(std::string &output, std::string &error);

private:
    struct Impl;
    Impl *impl;
};
