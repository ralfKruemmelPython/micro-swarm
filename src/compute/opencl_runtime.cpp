#include "opencl_runtime.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

#ifndef MICRO_SWARM_OPENCL
#define MICRO_SWARM_OPENCL 0
#endif

#ifndef MICRO_SWARM_OPENCL_DYNAMIC
#define MICRO_SWARM_OPENCL_DYNAMIC 0
#endif

#if MICRO_SWARM_OPENCL
#include <CL/cl.h>

#if MICRO_SWARM_OPENCL_DYNAMIC
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace {
const char *cl_err_to_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string load_kernel_source(const std::vector<std::string> &paths) {
    for (const auto &p : paths) {
        std::string src = read_file(p);
        if (!src.empty()) {
            return src;
        }
    }
    return "";
}

#if MICRO_SWARM_OPENCL_DYNAMIC
struct OpenCLApi {
    bool loaded = false;
#if defined(_WIN32)
    HMODULE lib = nullptr;
#else
    void *lib = nullptr;
#endif

    decltype(&clGetPlatformIDs) clGetPlatformIDs_fn = nullptr;
    decltype(&clGetPlatformInfo) clGetPlatformInfo_fn = nullptr;
    decltype(&clGetDeviceIDs) clGetDeviceIDs_fn = nullptr;
    decltype(&clGetDeviceInfo) clGetDeviceInfo_fn = nullptr;
    decltype(&clCreateContext) clCreateContext_fn = nullptr;
    decltype(&clCreateCommandQueue) clCreateCommandQueue_fn = nullptr;
    decltype(&clCreateCommandQueueWithProperties) clCreateCommandQueueWithProperties_fn = nullptr;
    decltype(&clCreateProgramWithSource) clCreateProgramWithSource_fn = nullptr;
    decltype(&clBuildProgram) clBuildProgram_fn = nullptr;
    decltype(&clGetProgramBuildInfo) clGetProgramBuildInfo_fn = nullptr;
    decltype(&clCreateKernel) clCreateKernel_fn = nullptr;
    decltype(&clSetKernelArg) clSetKernelArg_fn = nullptr;
    decltype(&clCreateBuffer) clCreateBuffer_fn = nullptr;
    decltype(&clEnqueueWriteBuffer) clEnqueueWriteBuffer_fn = nullptr;
    decltype(&clEnqueueReadBuffer) clEnqueueReadBuffer_fn = nullptr;
    decltype(&clEnqueueNDRangeKernel) clEnqueueNDRangeKernel_fn = nullptr;
    decltype(&clFinish) clFinish_fn = nullptr;
    decltype(&clReleaseMemObject) clReleaseMemObject_fn = nullptr;
    decltype(&clReleaseKernel) clReleaseKernel_fn = nullptr;
    decltype(&clReleaseProgram) clReleaseProgram_fn = nullptr;
    decltype(&clReleaseCommandQueue) clReleaseCommandQueue_fn = nullptr;
    decltype(&clReleaseContext) clReleaseContext_fn = nullptr;

    bool load(std::string &error) {
#if defined(_WIN32)
        lib = LoadLibraryA("OpenCL.dll");
#else
        lib = dlopen("libOpenCL.so", RTLD_LAZY);
        if (!lib) {
            lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
        }
#endif
        if (!lib) {
            error = "OpenCL library not found";
            return false;
        }

        auto load_sym = [&](auto &fn, const char *name) {
#if defined(_WIN32)
            fn = reinterpret_cast<decltype(fn)>(GetProcAddress(lib, name));
#else
            fn = reinterpret_cast<decltype(fn)>(dlsym(lib, name));
#endif
            return fn != nullptr;
        };

        bool ok = true;
        ok &= load_sym(clGetPlatformIDs_fn, "clGetPlatformIDs");
        ok &= load_sym(clGetPlatformInfo_fn, "clGetPlatformInfo");
        ok &= load_sym(clGetDeviceIDs_fn, "clGetDeviceIDs");
        ok &= load_sym(clGetDeviceInfo_fn, "clGetDeviceInfo");
        ok &= load_sym(clCreateContext_fn, "clCreateContext");
        ok &= load_sym(clCreateCommandQueue_fn, "clCreateCommandQueue");
        load_sym(clCreateCommandQueueWithProperties_fn, "clCreateCommandQueueWithProperties");
        ok &= load_sym(clCreateProgramWithSource_fn, "clCreateProgramWithSource");
        ok &= load_sym(clBuildProgram_fn, "clBuildProgram");
        ok &= load_sym(clGetProgramBuildInfo_fn, "clGetProgramBuildInfo");
        ok &= load_sym(clCreateKernel_fn, "clCreateKernel");
        ok &= load_sym(clSetKernelArg_fn, "clSetKernelArg");
        ok &= load_sym(clCreateBuffer_fn, "clCreateBuffer");
        ok &= load_sym(clEnqueueWriteBuffer_fn, "clEnqueueWriteBuffer");
        ok &= load_sym(clEnqueueReadBuffer_fn, "clEnqueueReadBuffer");
        ok &= load_sym(clEnqueueNDRangeKernel_fn, "clEnqueueNDRangeKernel");
        ok &= load_sym(clFinish_fn, "clFinish");
        ok &= load_sym(clReleaseMemObject_fn, "clReleaseMemObject");
        ok &= load_sym(clReleaseKernel_fn, "clReleaseKernel");
        ok &= load_sym(clReleaseProgram_fn, "clReleaseProgram");
        ok &= load_sym(clReleaseCommandQueue_fn, "clReleaseCommandQueue");
        ok &= load_sym(clReleaseContext_fn, "clReleaseContext");

        if (!ok) {
            error = "OpenCL symbols missing";
            unload();
            return false;
        }
        loaded = true;
        return true;
    }

    void unload() {
        if (lib) {
#if defined(_WIN32)
            FreeLibrary(lib);
#else
            dlclose(lib);
#endif
        }
        lib = nullptr;
        loaded = false;
    }
};

OpenCLApi g_api;

#define OCL_CALL(fn) g_api.fn##_fn
#else
#define OCL_CALL(fn) fn
#endif
} // namespace

struct OpenCLRuntime::Impl {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel diffuse_kernel = nullptr;

    cl_mem phero_food_a = nullptr;
    cl_mem phero_food_b = nullptr;
    cl_mem phero_danger_a = nullptr;
    cl_mem phero_danger_b = nullptr;
    cl_mem molecules_a = nullptr;
    cl_mem molecules_b = nullptr;
    bool food_ping = true;
    bool danger_ping = true;
    bool molecules_ping = true;
    int width = 0;
    int height = 0;

    std::string device_info;

    void release_buffers() {
        if (phero_food_a) {
            OCL_CALL(clReleaseMemObject)(phero_food_a);
            phero_food_a = nullptr;
        }
        if (phero_food_b) {
            OCL_CALL(clReleaseMemObject)(phero_food_b);
            phero_food_b = nullptr;
        }
        if (phero_danger_a) {
            OCL_CALL(clReleaseMemObject)(phero_danger_a);
            phero_danger_a = nullptr;
        }
        if (phero_danger_b) {
            OCL_CALL(clReleaseMemObject)(phero_danger_b);
            phero_danger_b = nullptr;
        }
        if (molecules_a) {
            OCL_CALL(clReleaseMemObject)(molecules_a);
            molecules_a = nullptr;
        }
        if (molecules_b) {
            OCL_CALL(clReleaseMemObject)(molecules_b);
            molecules_b = nullptr;
        }
    }

    void release_all() {
        release_buffers();
        if (diffuse_kernel) {
            OCL_CALL(clReleaseKernel)(diffuse_kernel);
            diffuse_kernel = nullptr;
        }
        if (program) {
            OCL_CALL(clReleaseProgram)(program);
            program = nullptr;
        }
        if (queue) {
            OCL_CALL(clReleaseCommandQueue)(queue);
            queue = nullptr;
        }
        if (context) {
            OCL_CALL(clReleaseContext)(context);
            context = nullptr;
        }
    }
};

OpenCLRuntime::OpenCLRuntime() : impl(new Impl()) {}
OpenCLRuntime::~OpenCLRuntime() {
    if (impl) {
        impl->release_all();
        delete impl;
        impl = nullptr;
    }
#if MICRO_SWARM_OPENCL_DYNAMIC
    if (g_api.loaded) {
        g_api.unload();
    }
#endif
}

bool OpenCLRuntime::init(int platform_index, int device_index, std::string &error) {
#if MICRO_SWARM_OPENCL_DYNAMIC
    if (!g_api.loaded) {
        if (!g_api.load(error)) {
            return false;
        }
    }
#endif
    cl_uint platform_count = 0;
    cl_int err = OCL_CALL(clGetPlatformIDs)(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        error = std::string("clGetPlatformIDs failed: ") + cl_err_to_string(err);
        return false;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    err = OCL_CALL(clGetPlatformIDs)(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clGetPlatformIDs failed: ") + cl_err_to_string(err);
        return false;
    }
    if (platform_index < 0 || platform_index >= static_cast<int>(platforms.size())) {
        error = "Invalid OpenCL platform index";
        return false;
    }
    impl->platform = platforms[platform_index];

    cl_uint device_count = 0;
    err = OCL_CALL(clGetDeviceIDs)(impl->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
    if (err != CL_SUCCESS || device_count == 0) {
        error = std::string("clGetDeviceIDs failed: ") + cl_err_to_string(err);
        return false;
    }
    std::vector<cl_device_id> devices(device_count);
    err = OCL_CALL(clGetDeviceIDs)(impl->platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clGetDeviceIDs failed: ") + cl_err_to_string(err);
        return false;
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        error = "Invalid OpenCL device index";
        return false;
    }
    impl->device = devices[device_index];

    char device_name[256] = {};
    OCL_CALL(clGetDeviceInfo)(impl->device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    char platform_name[256] = {};
    OCL_CALL(clGetPlatformInfo)(impl->platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
    impl->device_info = std::string(platform_name) + " / " + device_name;

    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)impl->platform, 0};
    impl->context = OCL_CALL(clCreateContext)(props, 1, &impl->device, nullptr, nullptr, &err);
    if (!impl->context || err != CL_SUCCESS) {
        error = std::string("clCreateContext failed: ") + cl_err_to_string(err);
        return false;
    }

#if MICRO_SWARM_OPENCL_DYNAMIC
    if (OCL_CALL(clCreateCommandQueueWithProperties)) {
        impl->queue = OCL_CALL(clCreateCommandQueueWithProperties)(impl->context, impl->device, nullptr, &err);
    } else {
        impl->queue = OCL_CALL(clCreateCommandQueue)(impl->context, impl->device, 0, &err);
    }
#else
#ifdef CL_VERSION_2_0
    impl->queue = clCreateCommandQueueWithProperties(impl->context, impl->device, nullptr, &err);
#else
    impl->queue = clCreateCommandQueue(impl->context, impl->device, 0, &err);
#endif
#endif
    if (!impl->queue || err != CL_SUCCESS) {
        error = std::string("clCreateCommandQueue failed: ") + cl_err_to_string(err);
        return false;
    }

    return true;
}

bool OpenCLRuntime::build_kernels(std::string &error) {
    std::vector<std::string> paths = {
        "src/compute/kernels/diffuse.cl",
        "../src/compute/kernels/diffuse.cl",
        "../../src/compute/kernels/diffuse.cl",
        "compute/kernels/diffuse.cl",
        "kernels/diffuse.cl"
    };
    std::string source = load_kernel_source(paths);
    if (source.empty()) {
        error = "Kernel source not found (diffuse.cl)";
        return false;
    }
    const char *src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_int err = CL_SUCCESS;
    impl->program = OCL_CALL(clCreateProgramWithSource)(impl->context, 1, &src_ptr, &src_len, &err);
    if (!impl->program || err != CL_SUCCESS) {
        error = std::string("clCreateProgramWithSource failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clBuildProgram)(impl->program, 1, &impl->device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        OCL_CALL(clGetProgramBuildInfo)(impl->program, impl->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        OCL_CALL(clGetProgramBuildInfo)(impl->program, impl->device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        error = std::string("clBuildProgram failed: ") + cl_err_to_string(err) + "\n" + log;
        return false;
    }
    impl->diffuse_kernel = OCL_CALL(clCreateKernel)(impl->program, "diffuse_and_evaporate", &err);
    if (!impl->diffuse_kernel || err != CL_SUCCESS) {
        error = std::string("clCreateKernel failed: ") + cl_err_to_string(err);
        return false;
    }
    return true;
}

bool OpenCLRuntime::init_fields(const GridField &phero_food,
                                const GridField &phero_danger,
                                const GridField &molecules,
                                std::string &error) {
    if (phero_food.width <= 0 || phero_food.height <= 0) {
        error = "Invalid field size";
        return false;
    }
    if (phero_food.width != molecules.width || phero_food.height != molecules.height ||
        phero_food.width != phero_danger.width || phero_food.height != phero_danger.height) {
        error = "Field sizes must match";
        return false;
    }
    impl->release_buffers();
    impl->width = phero_food.width;
    impl->height = phero_food.height;
    size_t bytes = static_cast<size_t>(impl->width) * impl->height * sizeof(float);
    cl_int err = CL_SUCCESS;
    impl->phero_food_a = OCL_CALL(clCreateBuffer)(impl->context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!impl->phero_food_a || err != CL_SUCCESS) {
        error = std::string("clCreateBuffer phero_food_a failed: ") + cl_err_to_string(err);
        return false;
    }
    impl->phero_food_b = OCL_CALL(clCreateBuffer)(impl->context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!impl->phero_food_b || err != CL_SUCCESS) {
        error = std::string("clCreateBuffer phero_food_b failed: ") + cl_err_to_string(err);
        return false;
    }
    impl->phero_danger_a = OCL_CALL(clCreateBuffer)(impl->context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!impl->phero_danger_a || err != CL_SUCCESS) {
        error = std::string("clCreateBuffer phero_danger_a failed: ") + cl_err_to_string(err);
        return false;
    }
    impl->phero_danger_b = OCL_CALL(clCreateBuffer)(impl->context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!impl->phero_danger_b || err != CL_SUCCESS) {
        error = std::string("clCreateBuffer phero_danger_b failed: ") + cl_err_to_string(err);
        return false;
    }
    impl->molecules_a = OCL_CALL(clCreateBuffer)(impl->context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!impl->molecules_a || err != CL_SUCCESS) {
        error = std::string("clCreateBuffer molecules_a failed: ") + cl_err_to_string(err);
        return false;
    }
    impl->molecules_b = OCL_CALL(clCreateBuffer)(impl->context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (!impl->molecules_b || err != CL_SUCCESS) {
        error = std::string("clCreateBuffer molecules_b failed: ") + cl_err_to_string(err);
        return false;
    }
    impl->food_ping = true;
    impl->danger_ping = true;
    impl->molecules_ping = true;

    err = OCL_CALL(clEnqueueWriteBuffer)(impl->queue, impl->phero_food_a, CL_TRUE, 0, bytes, phero_food.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueWriteBuffer phero_food failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clEnqueueWriteBuffer)(impl->queue, impl->phero_danger_a, CL_TRUE, 0, bytes, phero_danger.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueWriteBuffer phero_danger failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clEnqueueWriteBuffer)(impl->queue, impl->molecules_a, CL_TRUE, 0, bytes, molecules.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueWriteBuffer molecules failed: ") + cl_err_to_string(err);
        return false;
    }
    return true;
}

bool OpenCLRuntime::upload_fields(const GridField &phero_food,
                                  const GridField &phero_danger,
                                  const GridField &molecules,
                                  std::string &error) {
    if (phero_food.width != impl->width || phero_food.height != impl->height) {
        error = "Host field size mismatch";
        return false;
    }
    size_t bytes = static_cast<size_t>(impl->width) * impl->height * sizeof(float);
    cl_mem food_current = impl->food_ping ? impl->phero_food_a : impl->phero_food_b;
    cl_mem danger_current = impl->danger_ping ? impl->phero_danger_a : impl->phero_danger_b;
    cl_mem m_current = impl->molecules_ping ? impl->molecules_a : impl->molecules_b;
    cl_int err = OCL_CALL(clEnqueueWriteBuffer)(impl->queue, food_current, CL_TRUE, 0, bytes, phero_food.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueWriteBuffer phero_food failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clEnqueueWriteBuffer)(impl->queue, danger_current, CL_TRUE, 0, bytes, phero_danger.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueWriteBuffer phero_danger failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clEnqueueWriteBuffer)(impl->queue, m_current, CL_TRUE, 0, bytes, molecules.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueWriteBuffer molecules failed: ") + cl_err_to_string(err);
        return false;
    }
    return true;
}

bool OpenCLRuntime::step_diffuse(const FieldParams &pheromone_params,
                                 const FieldParams &molecule_params,
                                 bool do_copyback,
                                 GridField &phero_food,
                                 GridField &phero_danger,
                                 GridField &molecules,
                                 std::string &error) {
    if (!impl->diffuse_kernel || !impl->queue) {
        error = "OpenCL runtime not initialized";
        return false;
    }
    size_t global[2] = {static_cast<size_t>(impl->width), static_cast<size_t>(impl->height)};
    auto run_kernel = [&](cl_mem in_buf, cl_mem out_buf, const FieldParams &params) -> bool {
        cl_int err = CL_SUCCESS;
        err |= OCL_CALL(clSetKernelArg)(impl->diffuse_kernel, 0, sizeof(cl_mem), &in_buf);
        err |= OCL_CALL(clSetKernelArg)(impl->diffuse_kernel, 1, sizeof(cl_mem), &out_buf);
        err |= OCL_CALL(clSetKernelArg)(impl->diffuse_kernel, 2, sizeof(int), &impl->width);
        err |= OCL_CALL(clSetKernelArg)(impl->diffuse_kernel, 3, sizeof(int), &impl->height);
        err |= OCL_CALL(clSetKernelArg)(impl->diffuse_kernel, 4, sizeof(float), &params.diffusion);
        err |= OCL_CALL(clSetKernelArg)(impl->diffuse_kernel, 5, sizeof(float), &params.evaporation);
        if (err != CL_SUCCESS) {
            error = std::string("clSetKernelArg failed: ") + cl_err_to_string(err);
            return false;
        }
        err = OCL_CALL(clEnqueueNDRangeKernel)(impl->queue, impl->diffuse_kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            error = std::string("clEnqueueNDRangeKernel failed: ") + cl_err_to_string(err);
            return false;
        }
        return true;
    };

    cl_mem food_in = impl->food_ping ? impl->phero_food_a : impl->phero_food_b;
    cl_mem food_out = impl->food_ping ? impl->phero_food_b : impl->phero_food_a;
    if (!run_kernel(food_in, food_out, pheromone_params)) {
        return false;
    }
    impl->food_ping = !impl->food_ping;

    cl_mem danger_in = impl->danger_ping ? impl->phero_danger_a : impl->phero_danger_b;
    cl_mem danger_out = impl->danger_ping ? impl->phero_danger_b : impl->phero_danger_a;
    if (!run_kernel(danger_in, danger_out, pheromone_params)) {
        return false;
    }
    impl->danger_ping = !impl->danger_ping;

    cl_mem m_in = impl->molecules_ping ? impl->molecules_a : impl->molecules_b;
    cl_mem m_out = impl->molecules_ping ? impl->molecules_b : impl->molecules_a;
    if (!run_kernel(m_in, m_out, molecule_params)) {
        return false;
    }
    impl->molecules_ping = !impl->molecules_ping;

    if (do_copyback) {
        return copyback(phero_food, phero_danger, molecules, error);
    }
    return true;
}

bool OpenCLRuntime::copyback(GridField &phero_food, GridField &phero_danger, GridField &molecules, std::string &error) {
    if (phero_food.width != impl->width || phero_food.height != impl->height) {
        error = "Host field size mismatch";
        return false;
    }
    size_t bytes = static_cast<size_t>(impl->width) * impl->height * sizeof(float);
    cl_mem food_current = impl->food_ping ? impl->phero_food_a : impl->phero_food_b;
    cl_mem danger_current = impl->danger_ping ? impl->phero_danger_a : impl->phero_danger_b;
    cl_mem m_current = impl->molecules_ping ? impl->molecules_a : impl->molecules_b;
    cl_int err = OCL_CALL(clEnqueueReadBuffer)(impl->queue, food_current, CL_TRUE, 0, bytes, phero_food.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueReadBuffer phero_food failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clEnqueueReadBuffer)(impl->queue, danger_current, CL_TRUE, 0, bytes, phero_danger.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueReadBuffer phero_danger failed: ") + cl_err_to_string(err);
        return false;
    }
    err = OCL_CALL(clEnqueueReadBuffer)(impl->queue, m_current, CL_TRUE, 0, bytes, molecules.data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clEnqueueReadBuffer molecules failed: ") + cl_err_to_string(err);
        return false;
    }
    return true;
}

bool OpenCLRuntime::is_available() const {
    return impl && impl->context && impl->queue && impl->diffuse_kernel;
}

std::string OpenCLRuntime::device_info() const {
    if (!impl) {
        return "";
    }
    return impl->device_info;
}

bool OpenCLRuntime::print_devices(std::string &output, std::string &error) {
#if MICRO_SWARM_OPENCL_DYNAMIC
    if (!g_api.loaded) {
        if (!g_api.load(error)) {
            return false;
        }
    }
#endif
    cl_uint platform_count = 0;
    cl_int err = OCL_CALL(clGetPlatformIDs)(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        error = std::string("clGetPlatformIDs failed: ") + cl_err_to_string(err);
        return false;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    err = OCL_CALL(clGetPlatformIDs)(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        error = std::string("clGetPlatformIDs failed: ") + cl_err_to_string(err);
        return false;
    }
    std::ostringstream ss;
    for (cl_uint p = 0; p < platform_count; ++p) {
        char pname[256] = {};
        OCL_CALL(clGetPlatformInfo)(platforms[p], CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
        ss << "Platform " << p << ": " << pname << "\n";
        cl_uint device_count = 0;
        err = OCL_CALL(clGetDeviceIDs)(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        if (err != CL_SUCCESS || device_count == 0) {
            ss << "  (no devices)\n";
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        err = OCL_CALL(clGetDeviceIDs)(platforms[p], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            ss << "  (device query failed)\n";
            continue;
        }
        for (cl_uint d = 0; d < device_count; ++d) {
            char dname[256] = {};
            OCL_CALL(clGetDeviceInfo)(devices[d], CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
            ss << "  Device " << d << ": " << dname << "\n";
        }
    }
    output = ss.str();
    return true;
}

#else
OpenCLRuntime::OpenCLRuntime() : impl(nullptr) {}
OpenCLRuntime::~OpenCLRuntime() {}
bool OpenCLRuntime::init(int, int, std::string &error) { error = "OpenCL disabled at build time"; return false; }
bool OpenCLRuntime::build_kernels(std::string &error) { error = "OpenCL disabled at build time"; return false; }
bool OpenCLRuntime::init_fields(const GridField &, const GridField &, const GridField &, std::string &error) { error = "OpenCL disabled at build time"; return false; }
bool OpenCLRuntime::upload_fields(const GridField &, const GridField &, const GridField &, std::string &error) { error = "OpenCL disabled at build time"; return false; }
bool OpenCLRuntime::step_diffuse(const FieldParams &, const FieldParams &, bool, GridField &, GridField &, GridField &, std::string &error) { error = "OpenCL disabled at build time"; return false; }
bool OpenCLRuntime::copyback(GridField &, GridField &, GridField &, std::string &error) { error = "OpenCL disabled at build time"; return false; }
bool OpenCLRuntime::is_available() const { return false; }
std::string OpenCLRuntime::device_info() const { return ""; }
bool OpenCLRuntime::print_devices(std::string &, std::string &error) { error = "OpenCL disabled at build time"; return false; }
#endif
