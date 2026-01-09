cmake_minimum_required(VERSION 3.20)
project(micro_swarm LANGUAGES CXX)

option(MICRO_SWARM_OPENCL "Enable OpenCL support" ON)
option(MICRO_SWARM_OPENCL_DYNAMIC "Use dynamic OpenCL loading" OFF)

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
    src/compute/opencl_runtime.cpp
    src/compute/opencl_runtime.h
)

target_include_directories(micro_swarm PRIVATE src)

if (MICRO_SWARM_OPENCL)
    find_package(OpenCL QUIET)
    if (OpenCL_FOUND)
        target_compile_definitions(micro_swarm PRIVATE MICRO_SWARM_OPENCL=1)
        target_include_directories(micro_swarm PRIVATE ${OpenCL_INCLUDE_DIRS})
        if (MICRO_SWARM_OPENCL_DYNAMIC)
            target_compile_definitions(micro_swarm PRIVATE MICRO_SWARM_OPENCL_DYNAMIC=1)
        else()
            target_link_libraries(micro_swarm PRIVATE OpenCL::OpenCL)
        endif()
    else()
        target_compile_definitions(micro_swarm PRIVATE MICRO_SWARM_OPENCL=0)
    endif()
else()
    target_compile_definitions(micro_swarm PRIVATE MICRO_SWARM_OPENCL=0)
endif()
