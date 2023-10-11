#include "ConvolutionCommon.h"

namespace SNN
{

    ConvolutionCommon::ConvolutionCommon()
    {
    }

    ConvolutionCommon::~ConvolutionCommon()
    {
    }

    std::pair<int, int> ConvolutionCommon::GetPadding(const std::shared_ptr<Tensor> tensor)
    {
        int mPadY = 0, mPadX = 0;
        // input_shape [B, H, W, C]
        // get same for padding size X, Y
        const std::vector<int> &inputShape = tensor->InputShape();
        const std::vector<int> &outputShape = tensor->OutputShape();
        const std::vector<int> &kernelShape = tensor->KernelShape();

        if (tensor->GetPaddingType() == kPaddingSame)
        {
            int kernelHeightSize = (kernelShape[2] - 1) * tensor->dilation(0) + 1;
            int kernelWidthSize = (kernelShape[3] - 1) * tensor->dilation(1) + 1;
            int padNeedPadHeight = (outputShape[1] - 1) * tensor->stride(0) + kernelHeightSize - inputShape[1];
            int padNeededWidth = (outputShape[2] - 1) * tensor->stride(1) + kernelWidthSize - inputShape[2];
            mPadY = padNeedPadHeight / 2;
            mPadX = padNeededWidth / 2;
            return std::make_pair(mPadY, mPadX);
        }
        return std::make_pair(mPadY, mPadX);
    }

} // SNN