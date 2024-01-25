#ifndef POOLEXECUTION_H
#define POOLEXECUTION_H
#include "Execution.h"
namespace SNN
{

    class PoolExecution : public Execution
    {
    public:
        PoolExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~PoolExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        std::vector<size_t> PoolLocalWS(const std::vector<size_t> &gws, const size_t maxWorkGroupSize);

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        // cl_mem *inputCLData, *outputCLData;
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        std::vector<size_t> mGWS{1, 1, 1};
        std::vector<size_t> mLWS{1, 1, 1, 1};
        std::vector<int> mStrides{1, 1};
        std::vector<int> mKernels{1, 1};
        std::vector<int> mPaddings{0, 0};
        std::vector<int> mDilations{1, 1};
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
    };
} // namespace SNN

#endif // POOLEXECUTION_H