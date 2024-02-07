#include "TensorUtils.h"

namespace SNN
{
    bool TensorUtils::regionIsFull(const std::vector<int> &inputShape, const std::vector<std::vector<int>> &inputSizes)
    {
        int size = 1;
        for (int i = 0; i < 4; ++i)
        {
            size *= inputShape[i];
        }
        int regionSize = 0;
        for (auto &inputSize : inputSizes)
            regionSize += inputSize[1] * inputSize[0] * inputSize[2];
        return regionSize == size;
    }
} // namespace SNN
