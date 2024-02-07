#ifndef DOUBLECONVEXECUTION_H
#define DOUBLECONVEXECUTION_H
#include "Execution.h"
namespace SNN
{
    class DoubleConvExecution : public Execution
    {
    public:
        explicit DoubleConvExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend);
        virtual ~DoubleConvExecution() = default;
        virtual bool onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors) override;
        // virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        int numTensors;
        std::string kernelName;

    private:
        std::set<std::string> mBuildOptions;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        std::vector<size_t> mGWS{1, 1};
        std::vector<size_t> mLWS{1, 1};
    };
} // namespace SNN

#endif // DOUBLECONVEXECUTION_H