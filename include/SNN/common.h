#ifndef COMMON_h
#define COMMON_h
#include "op_data.h"
typedef struct BackendConfig
{
    enum Type
    {
        CPU = 0,
        OpenCL,
    };

    enum PrecisionMode
    {
        Precision_FP32 = 0,
        Precision_FP16,
        Precision_INT8,
    };
    Type backendType = OpenCL;
    PrecisionMode precisionType = Precision_FP32;
} BackendConfig;

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
typedef enum
{
    DepthwiseConv = 0,
    Conv2D = 1,
    Concat = 2,
    Relu = 3,
    AVERAGE_POOL_2D = 4,
    MAX_POOL_2D = 5,
    RESIZE_NEAREST_NEIGHBOR = 6,
    CONCATENATION = 7,
    ADD = 8,
    SUB = 9,
    MUL = 10,
    REALDIV = 11,
    MINIMUM = 12,
    MAXIMUM = 13,
    GREATER = 14,
    LESS = 15,

} OpTypes;
#endif