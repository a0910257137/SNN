#include "OpenCLBackend.h"

namespace SNN
{
    OpenCLBackend::OpenCLBackend(bool enable_fp16)
    {
        this->enable_fp16 = enable_fp16;
        mCLRuntime = new OpenCLRuntime();
    }
    OpenCLBackend::~OpenCLBackend()
    {
        delete mCLRuntime;
    }
    void OpenCLBackend::ArrayToOpenCL(Tensor &inputs, bool half_precision)
    {
        if (half_precision)
        {
        }
        else
        {
            cl_image_format clImageFormat;
            cl_kernel ckKernel;
            clImageFormat.image_channel_order = CL_RGBA;
            clImageFormat.image_channel_data_type = CL_FLOAT;
            // if (inputs.op_type == Depthwise)
            // {
            //     DepthwiseConvExecution depthwise(inputs);
            // }
            // int outputChannel = inputs.dims[0];
            // int kernelHeight = inputs.dims[1];
            // int kernelWidth = inputs.dims[2];
            // int multiplierChannel = inputs.dims[3];
            // std::cout << outputChannel << std::endl;
            // std::cout << kernelHeight << std::endl;
            // std::cout << kernelWidth << std::endl;
            // std::cout << multiplierChannel << std::endl;
            // exit(1);
            // dims
            // dst = clCreateImage2D(mCLRuntime->cxGPUContext, CL_MEM_WRITE_ONLY, &clImageFormat, filterHeight * filterWidth * multiplierChannel, (outputChennel + 3) / 4, 0, NULL, &mCLRuntime->err);
            // oclCheckError(mCLRuntime->err, CL_SUCCESS);
        }
        // cl_half *h_idata_fp16 = (cl_half *)malloc(max_m_k_n * max_m_k_n * sizeof(cl_half));
        // cl_half *h_odata_fp16 = (cl_half *)malloc(max_m_k_n * max_m_k_n * sizeof(cl_half));
        // half_from_float(h_idata, h_idata_fp16, max_m_k_n * max_m_k_n, CL_HALF_RTE);
        // half_from_float(h_odata, h_odata_fp16, max_m_k_n * max_m_k_n, CL_HALF_RTE);
    }
}