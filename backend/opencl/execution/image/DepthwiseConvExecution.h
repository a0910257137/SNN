#ifndef DEPTHWISECONVEXECUTION_H
#define DEPTHWISECONVEXECUTION_H
#include <iostream>
#include "ConvBaseExecution.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{
    class DepthwiseConvExecution : public ConvBaseExecution
    {
    public:
        explicit DepthwiseConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        ~DepthwiseConvExecution();
        DepthwiseConvExecution(const DepthwiseConvExecution &) = delete;
        DepthwiseConvExecution &operator=(const DepthwiseConvExecution &) = delete;
        bool onResize(std::shared_ptr<Tensor> tensor);
        bool onExecute();

    private:
        OpenCLBackend *mbackend;

    private:
        std::vector<size_t> mLWS{0, 0, 0, 0};
        std::vector<size_t> mGWS{0, 0, 0, 0};
        std::vector<int> mStrides{1, 1};
        std::vector<int> mPaddings{1, 1};
        std::vector<int> mDilations{1, 1};
        std::vector<size_t> mGlobalWorkSize{1, 1};
        std::vector<size_t> mLocalWorkSize{1, 1};
        cl_kernel mKernel;
        size_t mMaxWorkGroupSize;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        ImageBufferConverter *mImageConvert;
    };
}
#endif // DEPTHWISECONVEXECUTION_h