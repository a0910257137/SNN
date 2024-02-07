#ifndef CONVBASEBUFEXECUTION_H
#define CONVBASEBUFEXECUTION_H
#include "../image/Execution.h"
#include "backend/opencl/utils/coreUtils.h"
namespace SNN
{
    class ConvBaseBufExecution : public Execution
    {
    public:
        ConvBaseBufExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~ConvBaseBufExecution() = default;
    };
} // SNN
#endif // CONVBASEBUFEXECUTION_H