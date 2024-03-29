#ifndef UTILS_H
#define UTILS_H
#include "include/SNN/common.h"
#include "include/SNN/SNNDefine.h"
#include "include/SNN/macro.h"
#include "backend/opencl/core/OpenCLSetting.h"
#include "backend/opencl/core/runtime/OpenCLRuntime.h"
namespace SNN
{
    void getImageShape(const std::vector<int> &shape, const OpenCLBufferFormat type, size_t *imageShape);
    void CopyBufferToImage(OpenCLRuntime *runtime, const cl_mem &buffer, const cl_mem &image, int *w, int *h, cl_int &err);
    std::vector<int> TensorShapeFormat(const std::vector<int> &shape, DataFormat data_format);
}
#endif // UTILS_H