#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H
#include <string>
#include "include/SNN/common.h"
#include "NonCopyable.h"
#include <vector>
#include <iostream>
#include <tuple>
#include "misc/utils.h"
#include "misc/nms.h"
namespace SNN
{

    class PostProcessor : public NonCopyable
    {
    public:
        PostProcessor(ModelConfig &model_cfg);
        ~PostProcessor() = default;
        PostProcessor(const PostProcessor &) = delete;
        PostProcessor &operator=(const PostProcessor &) = delete;
        void MTFDConvProcessor(std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> &results,
                               int i,
                               float *resizedRatios,
                               float *cls_x,
                               float *bbox_x,
                               float *x,
                               const std::vector<int> &xShape,
                               const std::vector<int> &bbox_xShape,
                               const std::vector<int> &cls_xShape);
        void MTFDBaseProcessor(std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> &results,
                               int i,
                               float *resizedRatios,
                               float *cls_x,
                               float *bbox_x,
                               float *param_x,
                               float *trans_x);

    private:
        std::vector<std::vector<float>> FindCoord(
            float *cls_x,
            const std::vector<int> &cls_xShape,
            bool needSigmoid = false);

        std::vector<std::vector<float>> GetBboxWithConv(int i,
                                                        float *bbox_x,
                                                        std::vector<std::vector<float>> &coords,
                                                        const std::vector<int> &bbox_xShape);

        std::pair<std::vector<float>, std::vector<float>> GetParamsKpsWithConv(int stride_idx,
                                                                               float *bbox_x,
                                                                               std::vector<float> &coords,
                                                                               const std::vector<int> &xShape);

        std::vector<std::vector<float>> GetBbox(int stride_idx,
                                                float *bbox_x,
                                                std::vector<std::vector<float>> &coords,
                                                const std::vector<int> &bbox_xShape);

        std::pair<std::vector<float>, std::vector<float>> GetParamsKps(int stride_idx,
                                                                       std::vector<float> &coord,
                                                                       float *param_x,
                                                                       float *trans_x);

        std::vector<std::vector<float>> GetLandmarks(std::vector<float> &param);
        void ResizedBboxLandmarks(float *resizedRatios,
                                  std::vector<float> &bbox,
                                  std::vector<std::vector<float>> &landmarks,
                                  std::vector<float> &tran);

    private:
        float thresholdVal;
        ModelConfig model_cfg;
        std::map<std::string, std::tuple<int, float *>> headWeights;
        std::map<std::string, std::tuple<int, float *>> BFM;
        std::vector<int> meshX40;
        std::vector<int> meshY40;
        std::vector<int> meshX20;
        std::vector<int> meshY20;
        std::vector<int> meshX10;
        std::vector<int> meshY10;
        std::vector<int> strideFPN{8, 16, 32};
        int grid3x3[3][3][2] = {
            {{-1, -1}, {-1, 0}, {-1, 1}},
            {{0, -1}, {0, 0}, {0, 1}},
            {{1, -1}, {1, 0}, {1, 1}}};
        int grid5x5[5][5][2] = {
            {{-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2}},
            {{-1, -2}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}},
            {{0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2}},
            {{1, -2}, {1, -1}, {1, 0}, {1, 1}, {1, 2}},
            {{2, -2}, {2, -1}, {2, 0}, {2, 1}, {2, 2}}};
        std::vector<std::vector<std::string>> pred_bboxKeys{
            {"weight_pred_bbox_0",
             "bias_pred_bbox_0",
             "scale_bbox_0:0"},
            {"weight_pred_bbox_1",
             "bias_pred_bbox_1",
             "scale_bbox_1:0"},
            {"weight_pred_bbox_2",
             "bias_pred_bbox_2",
             "scale_bbox_2:0"}};
        std::vector<std::vector<std::string>> conv3x3_kpsKeys{
            {"weight_conv3x3_kps_0_0",
             "bias_conv3x3_kps_0_0",
             "weight_conv3x3_kps_0_1",
             "bias_conv3x3_kps_0_1"},
            {"weight_conv3x3_kps_1_0",
             "bias_conv3x3_kps_1_0",
             "weight_conv3x3_kps_1_1",
             "bias_conv3x3_kps_1_1"},
            {"weight_conv3x3_kps_2_0",
             "bias_conv3x3_kps_2_0",
             "weight_conv3x3_kps_2_1",
             "bias_conv3x3_kps_2_1"}};
        std::vector<std::vector<std::string>> pred_kpsKeys{
            {"weight_pred_kp_0",
             "bias_pred_kp_0"},
            {"weight_pred_kp_1",
             "bias_pred_kp_1"},
            {"weight_pred_kp_2",
             "bias_pred_kp_2"}};
        std::vector<std::vector<std::string>> conv3x3_paramsKeys{
            {"weight_conv3x3_params_0_0",
             "bias_conv3x3_params_0_0",
             "weight_conv3x3_params_0_1",
             "bias_conv3x3_params_0_1"},
            {"weight_conv3x3_params_1_0",
             "bias_conv3x3_params_1_0",
             "weight_conv3x3_params_1_1",
             "bias_conv3x3_params_1_1"},
            {"weight_conv3x3_params_2_0",
             "bias_conv3x3_params_2_0",
             "weight_conv3x3_params_2_1",
             "bias_conv3x3_params_2_1"}};
        std::vector<std::vector<std::string>> pred_paramsKeys{
            {"weight_pred_params_0",
             "bias_pred_params_0"},
            {"weight_pred_params_1",
             "bias_pred_params_1"},
            {"weight_pred_params_2",
             "bias_pred_params_2"}};
    };
}
#endif // POSTPROCESSOR_H