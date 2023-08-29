#ifndef DEPTHWISECONVEXECUTION_H
#define DEPTHWISECONVEXECUTION_H
#include <iostream>
#include "include/SNN/common.h"
#include "core/ConvolutionCommon.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{
    class DepthwiseConvExecution
    {
    public:
        explicit DepthwiseConvExecution(OpenCLBackend *mbackend);
        ~DepthwiseConvExecution();
        DepthwiseConvExecution(const DepthwiseConvExecution &) = delete;
        DepthwiseConvExecution &operator=(const DepthwiseConvExecution &) = delete;
        bool onInit(Tensor *inputs);
        bool onResize(Tensor *inputs, int *input_shape, int *output_shape);
        bool onExecute();

    private:
        OpenCLRuntime *mCLRuntime;
        size_t mLWS[4] = {0, 0, 0, 0};
        size_t mGWS[4] = {0, 0, 0, 0};
        int mStrides[2] = {1, 1};
        int mPaddings[2] = {0, 0};
        int mDilations[2] = {1, 1};
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        uint32_t mGlobalWorkSize[2] = {1, 1};
        uint32_t mLocalWorkSize[2] = {1, 1};
        // std::shared_ptr<DepthwiseConvParams> params;

        DepthwiseConvParams *params;
        // ConvolutionCommon *mCommon;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
    };
}
#endif // DEPTHWISECONVEXECUTION_h