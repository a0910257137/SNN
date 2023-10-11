#include "include/SNN/Tensor.h"
#define MAX_TENSOR_DIM 8
using namespace std;
namespace SNN
{
    Tensor::Tensor(int dimSize, DimensionType type)
    {
        SNN_ASSERT(dimSize <= MAX_TENSOR_DIM);
        mBuffer.dimensions = dimSize;
        mBuffer.type = halide_type_of<float>();
        mBuffer.device = 0;
        mBuffer.weightBytes = 0;
        mBuffer.biasBytes = 0;
    }
    Tensor::~Tensor()
    {
    }
    // void Tensor::SetType(int type)
    // {
    //     switch (type)
    //     {
    //         switch (type)
    //         {
    //         case DataType_DT_DOUBLE:
    //         case DataType_DT_FLOAT:
    //             mBuffer.type = halide_type_of<float>();
    //             break;
    //         case DataType_DT_BFLOAT16:
    //             mBuffer.type = halide_type_t(halide_type_float, 16);
    //             break;
    //         case DataType_DT_QINT32:
    //         case DataType_DT_INT32:
    //         case DataType_DT_BOOL:
    //         case DataType_DT_INT64:
    //             mBuffer.type = halide_type_of<int32_t>();
    //             break;
    //         case DataType_DT_QINT8:
    //         case DataType_DT_INT8:
    //             mBuffer.type = halide_type_of<int8_t>();
    //             break;
    //         case DataType_DT_QUINT8:
    //         case DataType_DT_UINT8:
    //             mBuffer.type = halide_type_of<uint8_t>();
    //             break;
    //         case DataType_DT_QUINT16:
    //         case DataType_DT_UINT16:
    //             mBuffer.type = halide_type_of<uint16_t>();
    //             break;
    //         case DataType_DT_QINT16:
    //         case DataType_DT_INT16:
    //             mBuffer.type = halide_type_of<int16_t>();
    //             break;
    //         default:
    //             printf("Unsupported data type!");
    //             SNN_ASSERT(false);
    //             break;
    //         }
    //     }
    // }
    void Tensor::SetMemoryPtr(int size)
    {
        mBuffer.hostPtr = std::vector<uint8_t>(size, 0);
    }
    const std::vector<int> Tensor::InputShape() const
    {
        std::vector<int> result;
        for (int i = 0; i < mBuffer.dimensions; ++i)
        {
            result.push_back(mBuffer.inputShapes[i]);
        }
        return result;
    }

    const std::vector<int> Tensor::OutputShape() const
    {
        std::vector<int> result;
        for (int i = 0; i < mBuffer.dimensions; ++i)
        {
            result.push_back(mBuffer.outputShapes[i]);
        }
        return result;
    }
    const std::vector<int> Tensor::KernelShape() const
    {
        std::vector<int> result;
        for (int i = 0; i < mBuffer.dimensions; ++i)
        {
            result.push_back(mBuffer.kernelShapes[i]);
        }
        return result;
    }

} // namespace SNN
