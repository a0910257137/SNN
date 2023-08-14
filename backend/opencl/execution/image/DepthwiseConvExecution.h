#ifndef DEPTHWISECONVEXECUTION_H
#define DEPTHWISECONVEXECUTION_H
#include <iostream>
#include "include/SNN/common.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "CL/opencl.h"
namespace SNN
{
    class DepthwiseConvExecution
    {
    public:
        DepthwiseConvExecution(Tensor *inputs);
        ~DepthwiseConvExecution();
        DepthwiseConvExecution(const DepthwiseConvExecution &) = delete;
        DepthwiseConvExecution &operator=(const DepthwiseConvExecution &) = delete;

    private:
        const float *mParams;
        int *mStrides;
        int *mPaddings;
        int *mDillations;
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        uint32_t *mGlobalWorkSize;
        uint32_t *mLocalWorkSize;
    };
}
#endif // DEPTHWISECONVEXECUTION_h