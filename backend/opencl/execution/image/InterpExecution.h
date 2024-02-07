#ifndef INTERPEXECUTION_H
#define INTERPEXECUTION_H
#include "Execution.h"
namespace SNN
{
    class InterpExecution : public Execution
    {
    public:
        InterpExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~InterpExecution() = default;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mOpenCLBackend;

    private:
        // cl_mem *inputCLData = nullptr, *outputCLData = nullptr;
        cl_mem inputCLData = NULL, outputCLData = NULL;

    private:
        cl_kernel mKernel;
        std::vector<size_t> mLWS{0, 0, 0, 0};
        std::vector<size_t> mGWS{0, 0, 0, 0};
        size_t mMaxWorkGroupSize;
        float mCordTransform[4];
    };
}

#endif // INTERPEXECUTION_H