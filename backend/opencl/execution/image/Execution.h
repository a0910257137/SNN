#ifndef EXECUTION_H
#define EXECUTION_H
// #include "include/SNN/common.h"
#include "include/SNN/Tensor.h"
#include "include/SNN/macro.h"
#include "core/BaseProctocol.h"
#include "core/ConvolutionCommon.h"
#include "backend/opencl/core/OpenCLBackend.h"

namespace SNN
{

    class Execution : public BaseProtocol
    {
    public:
        /**
         * @brief initializer.
         * @params backend that execution run on
         */
        Execution(OpenCLBackend *mbackend);
        virtual ~Execution() = default;
        virtual bool onResize(Tensor *inputs, int *input_shape, int *output_shape);
        virtual bool onExecution(Tensor *inputs);

    protected:
        OpenCLRuntime *mOpenCLRuntime;
    };
} // SNN
#endif // EXECUTION_H