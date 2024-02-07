#ifndef SEPERABLECONVEXECUTION_H
#define SEPERABLECONVEXECUTION_H
#include "ConvBaseExecution.h"
#include "backend/opencl/core/OpenCLBackend.h"

namespace SNN
{
    class SeperableConvExecution : public Execution
    {
    public:
        explicit SeperableConvExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend);
        virtual ~SeperableConvExecution() = default;
        SeperableConvExecution(const SeperableConvExecution &) = delete;
        SeperableConvExecution &operator=(const SeperableConvExecution &) = delete;
        virtual bool onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

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
}
#endif // SeperableConvExecution_h