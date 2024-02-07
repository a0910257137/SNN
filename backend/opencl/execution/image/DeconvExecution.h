#ifndef DECONVEXECUTION_H
#define DECONVEXECUTION_H
#include "Execution.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{

    class DeconvExecution : public Execution
    {
    public:
        DeconvExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~DeconvExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs) override;

    private:
        OpenCLBackend *mOpenCLBackend;

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

#endif // DECONVEXECUTION_H