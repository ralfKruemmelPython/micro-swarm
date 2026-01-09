#include "opencl_loader.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

OpenCLStatus probe_opencl() {
    OpenCLStatus status;
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA("OpenCL.dll");
    if (handle) {
        status.available = true;
        status.message = "OpenCL.dll geladen";
        FreeLibrary(handle);
    } else {
        status.available = false;
        status.message = "OpenCL.dll nicht gefunden";
    }
#else
    void *handle = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("libOpenCL.so.1", RTLD_LAZY);
    }
    if (handle) {
        status.available = true;
        status.message = "OpenCL runtime gefunden";
        dlclose(handle);
    } else {
        status.available = false;
        status.message = "OpenCL runtime nicht gefunden";
    }
#endif
    return status;
}
