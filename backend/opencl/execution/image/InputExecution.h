#ifndef INPUTEXECUTION_H
#define INPUTEXECUTION_H
#include "Execution.h"
#include "backend/opencl/core/ImageBufferConverter.h"

namespace SNN
{
    class InputExecution : public Execution
    {
    public:
        explicit InputExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~InputExecution();
        InputExecution(const InputExecution &) = delete;
        InputExecution &operator=(const InputExecution &) = delete;
        virtual bool onResize(std::shared_ptr<Tensor> tensor);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs) override;

    private:
        OpenCLBackend *mbackend;

        // private:
        //     ImageBufferConverter *mImageConvert;
    };

} // namespace SNN

#endif