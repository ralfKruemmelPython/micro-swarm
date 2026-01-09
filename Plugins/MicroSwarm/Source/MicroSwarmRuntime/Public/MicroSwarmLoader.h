#pragma once

#include "CoreMinimal.h"
#include "HAL/PlatformProcess.h"

#include "micro_swarm_api.h"

class FMicroSwarmDLL
{
public:
    bool Load(const FString& DllPath);
    void Unload();
    bool IsLoaded() const;

    ms_handle_t* Create(const ms_config_t* cfg) const;
    void Destroy(ms_handle_t* h) const;
    int Step(ms_handle_t* h, int steps) const;

    void GetFieldInfo(ms_handle_t* h, ms_field_kind kind, int& OutW, int& OutH) const;
    bool CopyFieldOut(ms_handle_t* h, ms_field_kind kind, TArray<float>& OutValues) const;
    bool CopyFieldIn(ms_handle_t* h, ms_field_kind kind, const TArray<float>& Values) const;

    bool CopyFieldToFloatArray(ms_handle_t* h, ms_field_kind kind, TArray<float>& OutValues, int& OutW, int& OutH) const;

private:
    void* DllHandle = nullptr;

    using ms_create_fn = ms_handle_t* (*)(const ms_config_t* cfg);
    using ms_destroy_fn = void (*)(ms_handle_t* h);
    using ms_step_fn = int (*)(ms_handle_t* h, int steps);
    using ms_get_field_info_fn = void (*)(ms_handle_t* h, ms_field_kind kind, int* w, int* hgt);
    using ms_copy_field_out_fn = int (*)(ms_handle_t* h, ms_field_kind kind, float* dst, int dst_count);
    using ms_copy_field_in_fn = int (*)(ms_handle_t* h, ms_field_kind kind, const float* src, int src_count);

    ms_create_fn MsCreate = nullptr;
    ms_destroy_fn MsDestroy = nullptr;
    ms_step_fn MsStep = nullptr;
    ms_get_field_info_fn MsGetFieldInfo = nullptr;
    ms_copy_field_out_fn MsCopyFieldOut = nullptr;
    ms_copy_field_in_fn MsCopyFieldIn = nullptr;
};
