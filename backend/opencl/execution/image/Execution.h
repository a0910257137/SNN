#ifndef EXECUTION_H
#define EXECUTION_H
#include "include/SNN/Tensor.h"
#include "include/SNN/macro.h"
#include "core/NonCopyable.h"
#include "core/ConvolutionCommon.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/opencl/core/ImageBufferConverter.h"
namespace SNN
{

    class Execution : public NonCopyable
    {
    public:
        /**
         * @brief initializer.
         * @params backend that execution run on
         */
        Execution(OpenCLBackend *mbackend);
        virtual ~Execution();
        virtual bool onResize(Tensor *inputs, int *input_shape, int *output_shape);
        virtual bool onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs);
        float *onConvert(std::shared_ptr<Tensor> &input);

    public:
        cl_context *GPUcontext;
        cl_command_queue *commandQueue;
        cl_image_format clImageFormat;
        ImageBufferConverter *mImageConvert;

    protected:
        OpenCLRuntime *mOpenCLRuntime;
    };
} // SNN
#endif // EXECUTION_H