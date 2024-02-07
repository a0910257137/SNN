#ifndef TENSORUTILS_H
#define TENSORUTILS_H
#include "include/SNN/Tensor.h"
#include "core/NonCopyable.h"
namespace SNN
{
    /** tensor utils */
    class __attribute__((visibility("default"))) TensorUtils
    {
    public:
        static bool regionIsFull(const std::vector<int> &inputSahpe, const std::vector<std::vector<int>> &inputSizes);
    };
} // namespace SNN

#endif // TENSORUTILS_H