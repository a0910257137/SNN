#ifndef DEPTHWISECONVEXECUTION_H
#define DEPTHWISECONVEXECUTION_H
#include <iostream>
#include "ConvBaseExecution.h"
#include "backend/opencl/core/OpenCLBackend.h"
namespace SNN
{
    class DepthwiseConvExecution : public ConvBaseExecution
    {
    public:
        explicit DepthwiseConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~DepthwiseConvExecution() = default;
        DepthwiseConvExecution(const DepthwiseConvExecution &) = delete;
        DepthwiseConvExecution &operator=(const DepthwiseConvExecution &) = delete;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

    private:
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        std::vector<int> mStrides{1, 1};
        std::vector<int> mPaddings{1, 1};
        std::vector<int> mDilations{1, 1};
        std::vector<size_t> mGWS{1, 1};
        std::vector<size_t> mLWS{1, 1};
        cl_kernel mKernel;
        size_t mMaxWorkGroupSize;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        std::set<std::string> mBuildOptions;
    };
}
#endif // DEPTHWISECONVEXECUTION_h