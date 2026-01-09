cmake_minimum_required(VERSION 3.20)
project(micro_swarm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(micro_swarm
    src/main.cpp
    src/sim/agent.cpp
    src/sim/agent.h
    src/sim/dna_memory.cpp
    src/sim/dna_memory.h
    src/sim/environment.cpp
    src/sim/environment.h
    src/sim/fields.cpp
    src/sim/fields.h
    src/sim/mycel.cpp
    src/sim/mycel.h
    src/sim/params.h
    src/sim/rng.h
    src/sim/io.cpp
    src/sim/io.h
    src/sim/report.cpp
    src/sim/report.h
    src/compute/opencl_loader.cpp
    src/compute/opencl_loader.h
)

target_include_directories(micro_swarm PRIVATE src)
