#ifndef COMMON_h
#define COMMON_h
#include "op_data.h"
#include <string>
#define THREAD = 1
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

typedef struct ModelConfig
{
    std::string mtfd_path = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down/tflite/FP32.tflite";
    // std::string mtfd_path = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down/tflite/backbone.tflite";
    // std::string mtfd_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/con2d_/conv2d.tflite";
    std::string weight_path = "";

} ModelConfig;

#endif