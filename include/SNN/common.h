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
    std::string mtfd_path = "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/tflite/mtfd_FP32.tflite";
    std::string weight_root = "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/weights";
    // std::string mtfd_path = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down/tflite/backbone.tflite";
    std::string bbox_path = "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/weights/bbox/binary";
    std::string params_path = "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/weights/kps/binary";
    std::string kps_path = "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/weights/params/binary";
    std::string BFM_path = "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/BFM";
    int nObjs = 15;
    int inputSizeY = 320, inputSizeX = 320;
    int n_R = 9, n_t3d = 2, n_shp = 40, n_exp = 11;
    float thresholdVal = .5f;
    bool model_head_post = true;

} ModelConfig;

#endif