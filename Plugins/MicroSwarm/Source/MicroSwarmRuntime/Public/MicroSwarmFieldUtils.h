#pragma once

#include "CoreMinimal.h"

class UTexture2D;

class FMicroSwarmFieldUtils
{
public:
    static UTexture2D* CreateTextureFromField(const TArray<float>& Values, int Width, int Height);
    static bool UpdateTextureFromField(UTexture2D* Texture, const TArray<float>& Values, int Width, int Height);
};
