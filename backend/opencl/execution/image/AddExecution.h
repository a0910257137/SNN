#ifndef ADDEXECUTION_H
#define ADDEXECUTION_H
#include "ConvBaseExecution.h"
#include "backend/opencl/core/OpenCLBackend.h"

namespace SNN
{
    class AddExecution : public Execution
    {
    public:
        explicit AddExecution(std::vector<std::shared_ptr<Tensor>> &tensors, OpenCLBackend *mbackend);
        virtual ~AddExecution() = default;
        AddExecution(const AddExecution &) = delete;
        AddExecution &operator=(const AddExecution &) = delete;
        virtual bool onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors) override;
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

    private:
        cl_mem inputCLData1 = NULL, inputCLData2 = NULL, outputCLData = NULL;

    private:
        std::set<std::string> mBuildOptions;
        std::shared_ptr<ConvolutionCommon> mConvCommon;
        cl_kernel mKernel;
        uint32_t mMaxWorkGroupSize;
        std::vector<size_t> mGWS{1, 1, 1};
        std::vector<size_t> mLWS{1, 1, 1};
        std::string kernelName;
        float mCordTransform[4];
        int numTensors;
    };
}
#endif // AddConvExecution_h