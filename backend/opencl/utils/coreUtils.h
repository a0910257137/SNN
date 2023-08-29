#ifndef UTILS_H
#define UTILS_H
#include "include/SNN/common.h"
#include "include/SNN/SNNDefine.h"
#include "include/SNN/macro.h"
#include "backend/opencl/core/OpenCLSetting.h"
#include <iostream>
namespace SNN
{

    void getImageShape(const int *shape, const OpenCLBufferFormat type, size_t *imageShape);
}
#endif // UTILS_H