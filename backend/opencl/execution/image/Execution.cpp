#include "Execution.h"

namespace SNN
{
    Execution::Execution(OpenCLBackend *mbackend)
    {
        mOpenCLRuntime = mbackend->CLRuntime();
    }
    bool Execution::onResize(Tensor *inputs, int *input_shape, int *output_shape)
    {
        return true;
    }
    bool Execution::onExecution(Tensor *inputs)
    {
        return true;
    }

} // SNN