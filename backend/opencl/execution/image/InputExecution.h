#ifndef INPUTEXECUTION_H
#define INPUTEXECUTION_H
#include "Execution.h"
namespace SNN
{
    class InputExecution : public Execution
    {
    public:
        explicit InputExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~InputExecution() = default;
        InputExecution(const InputExecution &) = delete;
        InputExecution &operator=(const InputExecution &) = delete;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
        virtual bool onInputExecute(float *input_data, std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    private:
        OpenCLBackend *mbackend;

    private:
        int bufferSize;

    private:
        cl_kernel mKernel;
        cl_mem inputCLData = NULL, outputCLData = NULL;
        std::vector<size_t> mGWS{1, 1};
        std::vector<size_t> mLWS{1, 1};
    };

} // namespace SNN

#endif