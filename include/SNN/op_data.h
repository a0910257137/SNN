#ifndef OP_DATA_H
#define OP_DATA_H
#include <stdint.h>
typedef enum
{
    kActNone = 0,
    kActRelu = 1,
    kActReluN1To1 = 2, // min(max(-1, x), 1)
    kActRelu6 = 3,     // min(max(0, x), 6)
    kActTanh = 4,
    kActSignBit = 5,
    kActSigmoid = 6,
} FusedActivation;
typedef struct
{
    int width;
    int height;
    int width_offset;
    int height_offset;
} PaddingValues;
// Possible padding types (for convolutions)
typedef enum
{
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
} Padding;
typedef struct
{
    Padding padding;
    int stride_width;
    int stride_height;
    int filter_width;
    int filter_height;
    FusedActivation activation;
    struct
    {
        PaddingValues padding;
    } computed;
} PoolParams;
typedef struct
{
    // Parameters for CONV_2D version 1.
    Padding padding;
    int stride_width;
    int stride_height;
    int kernel_dims[4];
    int bias_dims[1];
    float *weights;
    float *bias;
    int weight_bytes;
    int bias_bytes;
    FusedActivation activation;
    // Parameters for CONV_2D version 2.
    // Note: Version 2 supports dilation values not equal to 1.
    int dilation_width_factor;
    int dilation_height_factor;
} ConvParams;
typedef struct
{
    // Parameters for DepthwiseConv version 1 or above.
    Padding padding;
    int stride_width;
    int stride_height;
    int kernel_dims[4]; // MHWI
    int bias_dims[1];
    float *weights;
    float *bias;
    int depth_multiplier;
    int weight_bytes, bias_bytes;
    FusedActivation activation;
    // Parameters for DepthwiseConv version 2 or above.
    int dilation_width_factor;
    int dilation_height_factor;
} DepthwiseConvParams;

#endif