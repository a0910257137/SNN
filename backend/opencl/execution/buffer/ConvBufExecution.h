#ifndef CONVBUFFEXECUTION_H
#define CONVBUFFEXECUTION_H
#include "ConvBaseBufExecution.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "core/ConvolutionCommon.h"
#include "RasterBufExecution.h"
#include "../image/Execution.h"
namespace SNN
{
    class ConvBufExecution : public ConvBaseBufExecution
    {
    public:
        explicit ConvBufExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~ConvBufExecution() = default;
        ConvBufExecution(const ConvBufExecution &) = delete;
        ConvBufExecution &operator=(const ConvBufExecution &) = delete;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        // virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        // virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        void setConv1x1WeightBuffer(int packCout, int packCin, int elementSize, const float *filterDataPtr);

    private:
        void _generateFilterConvertRegion(const std::vector<int> &inputShape) const;

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
        int mKernelHeight, mKernelWidth;
        int mInputChannel, mOutputChannel;
        std::string kernelName;
        bool mIsTurn = false;
        bool mConv1x1Opt = false;
        bool mUseLocalMem = false;
        bool mUseSubgroup = false;
        cl_mem mKernelBuffer;
        cl_mem mBiasBuffer;
        cl_int err;
        std::set<std::string> mBuildOptions;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        bool mWeightUseBuffer = false;
    };
} // namespace SNN

#endif