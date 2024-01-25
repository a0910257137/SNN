#ifndef STEMEXECUTION_H
#define STEMEXECUTION_h
#include "backend/opencl/core/OpenCLBackend.h"
#include "Execution.h"
namespace SNN
{
    class StemExecution : public Execution
    {
    public:
        explicit StemExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend);
        virtual ~StemExecution() = default;
        StemExecution(const StemExecution &) = delete;
        StemExecution &operator=(const StemExecution &) = delete;
        virtual bool onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors) override;
        // virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors);
        virtual bool onInputExecute(float *input_data, std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

    private:
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        int bufferSize;
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
#endif // STEMEXECUTION_h