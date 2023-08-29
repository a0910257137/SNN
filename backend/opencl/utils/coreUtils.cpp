#include "coreUtils.h"
namespace SNN
{

    void getImageShape(const int *shape, const OpenCLBufferFormat type, size_t *imageShape)
    {
        SNN_ASSERT(imageShape != nullptr);
        if (type == CONV2D_FILTER)
        {
            imageShape[0] = shape[0] * shape[2] * shape[3];
            imageShape[1] = UP_DIV(shape[1], 4);
        }
        else if (type == DW_CONV2D_FILTER)
        {
            imageShape[0] = shape[0] * shape[2] * shape[3];
            imageShape[1] = UP_DIV(shape[1], 4);
        }

        else if (type == NHWC_BUFFER || type == NCHW_BUFFER)
        {
            imageShape[0] = UP_DIV(shape[3], 4) * shape[2];
            imageShape[1] = shape[0] * shape[1];
        }
        else if (type == CONV2D1x1_OPT_FILTER)
        {
            imageShape[0] = UP_DIV(shape[1], 4);
            imageShape[1] = shape[2] * shape[3] * shape[0];
        }
        else
        {
            printf("type not supported !!! \n");
        }
    }
} // namespace  SNN
