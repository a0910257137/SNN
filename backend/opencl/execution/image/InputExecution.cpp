
#include "InputExecution.h"

namespace SNN
{
    InputExecution::InputExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        this->mbackend = mbackend;
        const std::vector<std::vector<int>> &inputShapes = tensor->InputShape();
        SNN_ASSERT(inputShapes.size() == 1);
        std::string buildOption = "";
        // mImageConvert = new ImageBufferConverter(mOpenCLRuntime);

        if (mOpenCLRuntime->isWeightCpuTransHalf() == false)
        {
            buildOption = "-DBUFFER_INP_FP32";
        }
        const std::vector<int> &inpuShape = inputShapes[0];
        bool status = mbackend->ConvertNHWCBufferToImage(inpuShape, tensor->data_format, false, false);
        exit(1);
    }
    InputExecution::~InputExecution()
    {
    }
    bool InputExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        return true;
    }
    bool InputExecution::onExecute(std::vector<std::shared_ptr<Tensor>> &inputs, std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        std::cout << "RUN Convolution" << std::endl;
        return true;
    }
} // namespace SNN
