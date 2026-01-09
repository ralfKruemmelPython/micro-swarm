#pragma once

#include <string>

struct OpenCLStatus {
    bool available = false;
    std::string message;
};

OpenCLStatus probe_opencl();
