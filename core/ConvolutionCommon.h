#ifndef CONVOLUTIONCOMMON_H
#define CONVOLUTIONCOMMON_H
#include "BaseProctocol.h"
#include <map>
#include <utility>
#include <vector>
#include <memory>
#include "include/SNN/op_data.h"
#include "include/SNN/Tensor.h"
namespace SNN
{
    class ConvolutionCommon : public BaseProtocol
    {

    public:
        ConvolutionCommon(/* args */);
        ~ConvolutionCommon();
        /**
         * @brief Get the padding size for X and Y image.
         * @param input_shape    input array [B, H, W, C]
         * @param output_shape   output array [B, H, W, C]
         * @param kernel_size    kernel_size array [B, Y, X, C]
         * @param dilations        dialtes array
         * @param padMode        padMode
         * @return Paddingsize (Y, X)
         */
        std::pair<int, int> GetPadding(const std::shared_ptr<Tensor> tensor);

    private:
        /* data */
    };

} // SNN
#endif // CONVOLUTIONCOMMON_H