#include "MicroSwarmLoader.h"

bool FMicroSwarmDLL::Load(const FString& DllPath)
{
    if (DllHandle)
        return true;

    DllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!DllHandle)
        return false;

    MsCreate = reinterpret_cast<ms_create_fn>(FPlatformProcess::GetDllExport(DllHandle, TEXT("ms_create")));
    MsDestroy = reinterpret_cast<ms_destroy_fn>(FPlatformProcess::GetDllExport(DllHandle, TEXT("ms_destroy")));
    MsStep = reinterpret_cast<ms_step_fn>(FPlatformProcess::GetDllExport(DllHandle, TEXT("ms_step")));
    MsGetFieldInfo = reinterpret_cast<ms_get_field_info_fn>(FPlatformProcess::GetDllExport(DllHandle, TEXT("ms_get_field_info")));
    MsCopyFieldOut = reinterpret_cast<ms_copy_field_out_fn>(FPlatformProcess::GetDllExport(DllHandle, TEXT("ms_copy_field_out")));
    MsCopyFieldIn = reinterpret_cast<ms_copy_field_in_fn>(FPlatformProcess::GetDllExport(DllHandle, TEXT("ms_copy_field_in")));

    if (!MsCreate || !MsDestroy || !MsStep || !MsGetFieldInfo || !MsCopyFieldOut || !MsCopyFieldIn)
    {
        Unload();
        return false;
    }

    return true;
}

void FMicroSwarmDLL::Unload()
{
    MsCreate = nullptr;
    MsDestroy = nullptr;
    MsStep = nullptr;
    MsGetFieldInfo = nullptr;
    MsCopyFieldOut = nullptr;
    MsCopyFieldIn = nullptr;

    if (DllHandle)
    {
        FPlatformProcess::FreeDllHandle(DllHandle);
        DllHandle = nullptr;
    }
}

bool FMicroSwarmDLL::IsLoaded() const
{
    return DllHandle != nullptr;
}

ms_handle_t* FMicroSwarmDLL::Create(const ms_config_t* cfg) const
{
    return MsCreate ? MsCreate(cfg) : nullptr;
}

void FMicroSwarmDLL::Destroy(ms_handle_t* h) const
{
    if (MsDestroy)
        MsDestroy(h);
}

int FMicroSwarmDLL::Step(ms_handle_t* h, int steps) const
{
    return MsStep ? MsStep(h, steps) : 0;
}

void FMicroSwarmDLL::GetFieldInfo(ms_handle_t* h, ms_field_kind kind, int& OutW, int& OutH) const
{
    OutW = 0;
    OutH = 0;
    if (MsGetFieldInfo)
        MsGetFieldInfo(h, kind, &OutW, &OutH);
}

bool FMicroSwarmDLL::CopyFieldOut(ms_handle_t* h, ms_field_kind kind, TArray<float>& OutValues) const
{
    if (!MsCopyFieldOut)
        return false;
    if (OutValues.Num() <= 0)
        return false;
    int rc = MsCopyFieldOut(h, kind, OutValues.GetData(), OutValues.Num());
    return rc > 0;
}

bool FMicroSwarmDLL::CopyFieldIn(ms_handle_t* h, ms_field_kind kind, const TArray<float>& Values) const
{
    if (!MsCopyFieldIn)
        return false;
    if (Values.Num() <= 0)
        return false;
    int rc = MsCopyFieldIn(h, kind, Values.GetData(), Values.Num());
    return rc > 0;
}

bool FMicroSwarmDLL::CopyFieldToFloatArray(ms_handle_t* h, ms_field_kind kind, TArray<float>& OutValues, int& OutW, int& OutH) const
{
    GetFieldInfo(h, kind, OutW, OutH);
    if (OutW <= 0 || OutH <= 0)
        return false;

    const int Count = OutW * OutH;
    OutValues.SetNumUninitialized(Count);
    return CopyFieldOut(h, kind, OutValues);
}
