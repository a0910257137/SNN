#ifndef COMMON_h
#define COMMON_h
#include <CL/cl.h>
#include "op_data.h"
typedef enum
{
    CPU = 0,
    OpenCL = 1,
} DeviceType;

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
    Conv = 1,
    Concat = 2,
    Relu = 3,
    AveragePooling2D = 4,
    MaxPooling2D = 5
} OpTypes;

typedef struct IntArray
{
    int size;

#if defined(_MSC_VER)
    // Context for why this is needed is in http://b/189926408#comment21
    int data[1];
#elif (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
       __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                             \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
    // gcc 6.1+ have a bug where flexible members aren't properly handled
    // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
    int data[0];
#else
    int data[];
#endif
} IntArray;
typedef struct Tensor
{
    size_t bytes;
    IntArray *dims;
    void *op_data;
    OpTypes op_type;
    DeviceType device_type;
    cl_mem image;

} Tensor;

typedef struct SNNNode
{
    // Inputs and Outputs to this node expressed as indices into the simulator's tensors.
    IntArray *inputs, *outputs;
    void *user_data;
    bool status;
    Tensor *tensor;
} SNNNode;
#endif