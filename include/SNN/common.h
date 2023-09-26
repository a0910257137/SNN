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
    AveragePooling2D = 4,
    MaxPooling2D = 5
} OpTypes;
#endif