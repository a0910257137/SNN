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

typedef enum
{
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
} Padding;
typedef enum
{
    DEPTHWISECONV2D = 0,
    CONV2D = 1,
    DECONV2D = 2,
    AVERAGE_POOL_2D = 3,
    MAX_POOL_2D = 4,
    RESIZE_NEAREST_NEIGHBOR = 5,
    Relu = 6,
    CONCATENATION = 7,
    ADD = 8,
    SUB = 9,
    MUL = 10,
    REALDIV = 11,
    MINIMUM = 12,
    MAXIMUM = 13,
    GREATER = 14,
    LESS = 15,
    SEPARABLE = 16,
    INPUTDATA = 17,
} OpTypes;

typedef enum
{
    SandGlassBlock = 0,

} MergedOpTypes;

typedef enum
{
    kNoType = 0,
    kFloat32 = 1,
    kInt32 = 2,
    kUInt8 = 3,
    kInt64 = 4,
    kString = 5,
    kBool = 6,
    kInt16 = 7,
    kComplex64 = 8,
    kInt8 = 9,
    kFloat16 = 10,
    kFloat64 = 11,
    kComplex128 = 12,
    kUInt64 = 13,
    kUInt32 = 14,
} DType;

#endif