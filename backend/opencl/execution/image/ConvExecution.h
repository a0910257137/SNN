#ifndef CONVEXECUTION_H
#define CONVEXECUTION_H
#include "ConvBaseExecution.h"
#include "Execution.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "core/ConvolutionCommon.h"

namespace SNN
{
    class ConvExecution : public ConvBaseExecution
    {
    public:
        explicit ConvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~ConvExecution() = default;
        ConvExecution(const ConvExecution &) = delete;
        ConvExecution &operator=(const ConvExecution &) = delete;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

    private:
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        std::vector<int> mStrides{1, 1};
        std::vector<int> mPaddings{0, 0};
        std::vector<int> mDilations{1, 1};
        std::vector<size_t> mGWS{1, 1, 1};
        std::vector<size_t> mLWS{1, 1, 1};
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        bool mIsTurn;
        bool mConv1x1Opt{false};
        bool mUseLocalMem{false};
        cl_mem mKernelBuffer;
        cl_mem mBiasBuffer;
        std::set<std::string> mBuildOptions;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        bool mWeightUseBuffer = false;
    };
} // namespace SNN

#endif