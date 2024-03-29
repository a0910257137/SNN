#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <stdlib.h>
#include "SNNDefine.h"
#include "HalideRuntime.h"
#include "TypeGenerated.h"
#include <memory>
namespace SNN
{
    class __attribute__((visibility("default"))) Tensor
    {
        enum DimensionType
        {
            TENSORFLOW,
            TORCH
        };

        enum MapType
        {
            MAP_TENSOR_WRITE = 0,
            MAP_TENSOR_READ = 1,
        };

    public:
        /**
         * @brief create a tensor with dimension size and type without acquire memory for data.
         * @param dimSize   dimension size.
         * @param type      dimension type.
         */
        Tensor(int dimSize = 4, DimensionType type = TENSORFLOW);

        /** deinitializer */
        ~Tensor();

    private:
        // remove all assignment operator
        Tensor(const Tensor &tensor) = delete;
        Tensor(const Tensor &&tensor) = delete;
        Tensor &operator=(const Tensor &) = delete;
        Tensor &operator=(const Tensor &&) = delete;

    public:
        const halide_buffer_t &buffer() const
        {
            return mBuffer;
        }
        halide_buffer_t &buffer()
        {
            return mBuffer;
        }

        /**
         * @brief get dimension type.
         * @return dimension type.
         */
        DimensionType getDimensionType() const;
        /**
         * @brief set data type.
         * @param type data type defined in 'Type_generated.h'.
         */
        void SetType(int type);
        /**
         * @brief get data type.
         * @return data type.
         */
        inline halide_type_t GetType() const
        {
            return mBuffer.type;
        }

        /**
         * @brief visit host memory, data type is represented by `T`.
         * @return data point in `T` type.
         */
        template <typename T>
        T *host() const
        {
            return (T *)mBuffer.hostPtr;
        }
        /**
         * @brief visit device memory.
         * @return device data ID. what the ID means varies between backends.
         */
        uint64_t deviceId() const
        {
            return mBuffer.device;
        }

    public:
        int dimensions() const
        {
            return mBuffer.dimensions;
        }

        /**
         * @brief Get input Tensor dimensions
         * @return Input Tensor dimensions.
         */
        const std::vector<std::vector<int>> InputShape() const;
        /**
         * @brief Get output Tensor dimensions
         * @return Output Tensor dimensions.
         */
        const std::vector<int> OutputShape() const;
        /**
         * @brief Get kernel shapes
         *
         * @return Output kernel shapes.
         */
        const std::vector<int> KernelShape() const;
        void SetMemoryPtr(int size);
        std::vector<int> inputIndex;
        std::vector<int> outputIndex;

    public:
        /**
         * Get member functions
         **/
        inline uint32_t batch(int index) const
        {
            return mBuffer.inputShapes[index][0];
        }
        inline uint32_t height(int index) const
        {
            return mBuffer.inputShapes[index][1];
        }
        inline uint32_t width(int index) const
        {
            return mBuffer.inputShapes[index][2];
        }
        inline uint32_t channel(int index) const
        {
            return mBuffer.inputShapes[index][3];
        }
        inline int stride(int index) const
        {
            return mBuffer.strides[index];
        }
        inline int dilation(int index) const
        {
            return mBuffer.dilations[index];
        }
        inline uint32_t weight_bytes() const
        {
            return mBuffer.weightBytes;
        }
        inline uint32_t bias_bytes() const
        {
            return mBuffer.biasBytes;
        }
        inline int GetOpType()
        {
            return mBuffer.opType;
        }
        inline int GetActType()
        {
            return mBuffer.actType;
        }
        inline int GetPaddingType()
        {
            return mBuffer.paddingType;
        }
        inline int GetConcatAxis() const
        {
            return mBuffer.axis;
        }
        inline cl_mem *GetDeviceInputData()
        {
            return &mBuffer.mDeviceBuffer.inputData;
        }
        inline cl_mem *GetDeviceOutputData()
        {
            return &mBuffer.mDeviceBuffer.outputData;
        }

        inline cl_mem GetDeviceFilter() const
        {
            return mBuffer.mDeviceBuffer.mFilter;
        }
        inline cl_mem GetDeviceBias() const
        {
            return mBuffer.mDeviceBuffer.mBias;
        }
        inline const std::shared_ptr<std::vector<std::pair<float *, float *>>> GetMainMemory() const
        {
            return mainMemory;
        }

        inline std::string GetOpName() const
        {
            return this->name;
        }
        inline const std::vector<uint8_t> &GetMemoryPtrIndex() const
        {
            return mBuffer.hostPtr;
        }
        // PtrIndex for point to mainMemory
        inline void SetMemoryPtrIndex(int index, uint8_t value)
        {
            mBuffer.hostPtr[index] = value;
        }
        /**
         *Set functions
         **/
        inline void SetPaddingType(int paddingType)
        {
            mBuffer.paddingType = paddingType;
        }
        inline void SetOpType(int value)
        {
            mBuffer.opType = value;
        }
        inline void SetActType(int value)
        {
            mBuffer.actType = value;
        }
        inline void SetConcatAxis(int value)
        {
            mBuffer.axis = value;
        }
        /**
         * Kernel information such as stride, shape, bytes and so on ...
         */
        inline void SetStride(int index, int stride)
        {
            mBuffer.strides[index] = stride;
        }
        inline void SetWeightBytes(uint32_t bytes)
        {
            mBuffer.weightBytes = bytes;
        }
        inline void SetBiasBytes(uint32_t bytes)
        {
            mBuffer.biasBytes = bytes;
        }
        inline void SetDilation(int index, int dil)
        {
            mBuffer.dilations[index] = dil;
        }
        inline void SetInputShape(std::vector<uint32_t> &shape)
        {
            mBuffer.inputShapes.push_back(shape);
        }
        inline void SetOutputShape(int index, uint32_t value)
        {
            mBuffer.outputShapes[index] = value;
        }
        inline void SetKernelShape(int index, int value)
        {
            mBuffer.kernelShapes[index] = value;
        }
        inline void SetBiasShape(int index, int value)
        {
            mBuffer.biasShapes[index] = value;
        }
        inline void SetDeviceInputData(cl_mem &data)
        {
            mBuffer.mDeviceBuffer.inputData = data;
        }
        inline void SetDeviceOutputData(cl_mem &data)
        {
            mBuffer.mDeviceBuffer.outputData = data;
        }
        inline void SetDeviceFilter(cl_mem &data)
        {
            mBuffer.mDeviceBuffer.mFilter = data;
        }
        inline void SetDeviceBias(cl_mem &data)
        {
            mBuffer.mDeviceBuffer.mBias = data;
        }
        inline void SetOpName(std::string &name)
        {
            this->name = name;
        }
        void SetMainMemory(std::shared_ptr<std::vector<std::pair<float *, float *>>> memory)
        {
            mainMemory = memory;
        }

    public:
        DataFormat data_format = DATA_FORMAT_NHWC;
        bool is_init = true;

    protected:
        std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory;
        std::string name;
        halide_buffer_t mBuffer;
    };
}
#endif
