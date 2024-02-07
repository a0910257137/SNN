#include "RasterBufExecution.h"
namespace SNN
{
    RasterBufExecution::RasterBufExecution(std::shared_ptr<Tensor> tensor, OpenCLBackend *mbackend) : Execution(mbackend)
    {
        const std::vector<int> &kernelSahpe = tensor->KernelShape();
        mOutputChannel = kernelSahpe[0], mInputChannel = kernelSahpe[1], mKernelHeight = kernelSahpe[2], mKernelWidth = kernelSahpe[3];
        this->mbackend = mbackend;
    }
    bool RasterBufExecution::onResize(std::shared_ptr<Tensor> tensor)
    {
        mTempInput = nullptr, mTempOutput = nullptr;
        std::vector<std::vector<int>> kernelSize = _generateFilterConvertRegion();
        const std::vector<int> &kernelSahpe = tensor->KernelShape();
        auto regionNum = kernelSize.size();

        mNeedZero = !TensorUtils::regionIsFull(tensor->KernelShape(), kernelSize);
        // Alloc Temp buffer
        auto bufferUnitSize = mOpenCLRuntime->isSupportedFP16() ? sizeof(cl_half) : sizeof(float);
        cl_int err = CL_SUCCESS;
        int bufferSize = mOutputChannel * mInputChannel * mKernelHeight * mKernelWidth * bufferUnitSize;
        mTempInput = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
        mTempOutput = clCreateBuffer(*GPUcontext, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
        int kernel_idx = 0;
        auto outputShape = TensorShapeFormat(kernelSahpe, DATA_FORMAT_NHWC);
        return true;
    }
    std::vector<std::vector<int>> RasterBufExecution::_generateFilterConvertRegion()
    {
        std::vector<std::vector<int>> kernelSize(4, std::vector<int>(3, 0));
        for (int so = 0; so < 4; ++so)
        {
            int oSize = (mOutputChannel - so + 3) / 4;
            if (oSize <= 0)
                continue;
            kernelSize[so][0] = oSize;
            kernelSize[so][1] = mInputChannel;
            kernelSize[so][2] = mKernelWidth * mKernelHeight;
        }
        return kernelSize;
    }

} // namespace SNN
