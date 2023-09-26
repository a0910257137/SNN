#ifndef CONVBASEEXECUTION_H
#define CONVBASEEXECUTION_H
#include "Execution.h"
#include "backend/opencl/utils/coreUtils.h"
namespace SNN
{
    class ConvBaseExecution : public Execution
    {
    public:
        ConvBaseExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend);
        virtual ~ConvBaseExecution();
        // protected:
        //     cl_mem mBias;
    };
} // SNN
#endif // DEPTHWISECONVEXECUTION_h