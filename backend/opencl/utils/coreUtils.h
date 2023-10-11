#ifndef UTILS_H
#define UTILS_H
#include "include/SNN/common.h"
#include "include/SNN/SNNDefine.h"
#include "include/SNN/macro.h"
#include "backend/opencl/core/OpenCLSetting.h"
#include "backend/opencl/core/runtime/OpenCLRuntime.h"
#include <iostream>
namespace SNN
{
    void getImageShape(const std::vector<int> &shape, const OpenCLBufferFormat type, size_t *imageShape);
    void CopyBufferToImage(OpenCLRuntime *runtime, const cl_mem &buffer, const cl_mem &image, int *w, int *h, cl_int &err);
    std::vector<int> TensorShapeFormat(const Tensor *input);
}
#endif // UTILS_H