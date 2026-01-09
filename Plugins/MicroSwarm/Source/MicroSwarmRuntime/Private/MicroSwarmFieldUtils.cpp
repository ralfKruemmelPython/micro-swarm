#include "MicroSwarmFieldUtils.h"

#include "Engine/Texture2D.h"

UTexture2D* FMicroSwarmFieldUtils::CreateTextureFromField(const TArray<float>& Values, int Width, int Height)
{
    if (Width <= 0 || Height <= 0 || Values.Num() != Width * Height)
        return nullptr;

    UTexture2D* Texture = UTexture2D::CreateTransient(Width, Height, PF_R32_FLOAT);
    if (!Texture)
        return nullptr;

    Texture->Filter = TF_Nearest;
    Texture->AddressX = TA_Clamp;
    Texture->AddressY = TA_Clamp;
    Texture->SRGB = false;

    UpdateTextureFromField(Texture, Values, Width, Height);
    return Texture;
}

bool FMicroSwarmFieldUtils::UpdateTextureFromField(UTexture2D* Texture, const TArray<float>& Values, int Width, int Height)
{
    if (!Texture || Width <= 0 || Height <= 0 || Values.Num() != Width * Height)
        return false;

    if (!Texture->GetPlatformData() || Texture->GetPlatformData()->Mips.Num() == 0)
        return false;

    FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
    void* Data = Mip.BulkData.Lock(LOCK_READ_WRITE);
    if (!Data)
    {
        Mip.BulkData.Unlock();
        return false;
    }

    const size_t ByteCount = sizeof(float) * static_cast<size_t>(Values.Num());
    FMemory::Memcpy(Data, Values.GetData(), ByteCount);
    Mip.BulkData.Unlock();

    Texture->UpdateResource();
    return true;
}
