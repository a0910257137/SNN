#ifndef EXECUTION_H
#define EXECUTION_H
#include "include/SNN/Tensor.h"
#include "include/SNN/macro.h"
#include "core/NonCopyable.h"
#include "core/ConvolutionCommon.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/opencl/core/ImageBufferConverter.h"
#include "backend/opencl/core/BufferConvertor.h"
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
        virtual bool onInputExecute(float *input_data, std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors);
        virtual bool onOptimizedResize(std::vector<std::shared_ptr<Tensor>> &tensors);
        virtual bool onOptimizedExecute(std::vector<std::shared_ptr<Tensor>> &input_tensors, std::vector<std::shared_ptr<Tensor>> &output_tensors);
        float *onConvert(std::shared_ptr<Tensor> &input);

    public:
        cl_context *GPUcontext;
        cl_command_queue *commandQueue;
        cl_image_format clImageFormat;
        ImageBufferConverter *mImageConvert;
        BufferConvertor *mBufferConvertor;

    protected:
        OpenCLRuntime *mOpenCLRuntime;
    };
} // SNN
#endif // EXECUTION_H