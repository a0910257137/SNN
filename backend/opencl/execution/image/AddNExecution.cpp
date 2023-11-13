#include "AddNExecution.h"

namespace SNN
{
    AddN::AddN(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
    }

    bool AddN::onResize(std::shared_ptr<Tensor> tensor)
    {
    }
    bool AddN::onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs)
    {
    }
} // namespace SNN
