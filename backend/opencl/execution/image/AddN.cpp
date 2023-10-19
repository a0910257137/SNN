#include "AddN.h"

namespace SNN
{
    AddN::AddN(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mOpenCLBackend = mbackend;
    }

    bool AddN::onResize(std::shared_ptr<Tensor> tensor)
    {

    }
    bool AddN::onExecute()
    {
    }
} // namespace SNN
