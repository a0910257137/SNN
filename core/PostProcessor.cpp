#include "PostProcessor.h"
namespace SNN
{

    PostProcessor::PostProcessor(ModelConfig &model_cfg)
    {
        this->thresholdVal = model_cfg.thresholdVal;
        meshX40 = std::vector<int>(40);
        meshY40 = std::vector<int>(40);
        meshX20 = std::vector<int>(20);
        meshY20 = std::vector<int>(20);
        meshX10 = std::vector<int>(10);
        meshY10 = std::vector<int>(10);
        std::iota(std::begin(meshX40), std::end(meshX40), 0); // 0 is the starting number
        std::iota(std::begin(meshY40), std::end(meshY40), 0); // 0 is the starting number
        std::iota(std::begin(meshX20), std::end(meshX20), 0); // 0 is the starting number
        std::iota(std::begin(meshY20), std::end(meshY20), 0); // 0 is the starting number
        std::iota(std::begin(meshX10), std::end(meshX10), 0); // 0 is the starting number
        std::iota(std::begin(meshY10), std::end(meshY10), 0); // 0 is the starting number

        GetWeights(model_cfg.bbox_path, headWeights);
        GetWeights(model_cfg.params_path, headWeights);
        GetWeights(model_cfg.kps_path, headWeights);
        GetWeights(model_cfg.BFM_path, BFM);
    }

    void PostProcessor::MTFDBaseProcessor(std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> &results,
                                          int i,
                                          float *resizedRatios,
                                          float *cls_x,
                                          float *bbox_x,
                                          float *param_x,
                                          float *trans_x)
    {
        std::vector<int> cls_xShape, bbox_xShape;
        if (i == 0)
        {
            cls_xShape = {1, 40, 40, 4};
            bbox_xShape = {1, 40, 40, 8};
        }

        else if (i == 1)
        {
            cls_xShape = {1, 20, 20, 4};
            bbox_xShape = {1, 20, 20, 8};
        }

        else if (i == 2)
        {
            cls_xShape = {1, 10, 10, 4};
            bbox_xShape = {1, 10, 10, 8};
        }

        std::vector<std::vector<float>> outputCoords = FindCoord(cls_x, cls_xShape, true);
        if (outputCoords.size() == 0)
            return;

        // x1, y1, x2, y2, score, c
        std::vector<std::vector<float>> bboxes = GetBbox(i, bbox_x, outputCoords, bbox_xShape);
        std::pair<int, std::vector<float>> nmsResults = NMS(bboxes, 0.5);
        int &selectedIdx = nmsResults.first;
        std::vector<float> &bbox = nmsResults.second;
        std::vector<float> outputCoord = outputCoords[selectedIdx];
        std::pair<std::vector<float>, std::vector<float>> ParamsKps = GetParamsKps(i, outputCoord, param_x, trans_x);
        std::vector<float> &param = ParamsKps.first;
        std::vector<float> &tran = ParamsKps.second;
        std::vector<std::vector<float>> landmarks = GetLandmarks(param);
        ResizedBboxLandmarks(resizedRatios, bbox, landmarks, tran);
        results.first.push_back(bbox);
        results.second.push_back(landmarks);
        // for (int i = 0; i < 68; i++)
        // {
        //     std::cout << "(" << landmarks[i][0] << ", " << landmarks[i][1] << ")" << std::endl;
        // }
        // exit(1);
    }

    void PostProcessor::MTFDConvProcessor(std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> &results,
                                          int i,
                                          float *resizedRatios,
                                          float *cls_x,
                                          float *bbox_x,
                                          float *x,
                                          const std::vector<int> &xShape,
                                          const std::vector<int> &bbox_xShape,
                                          const std::vector<int> &cls_xShape)
    {

        std::vector<std::vector<float>> outputCoords = FindCoord(cls_x, cls_xShape);
        if (outputCoords.size() == 0)
            return;
        // x1, y1, x2, y2, score, c
        std::vector<std::vector<float>> bboxes = GetBboxWithConv(i, bbox_x, outputCoords, bbox_xShape);
        // // select index, selected bbox
        std::pair<int, std::vector<float>> nmsResults = NMS(bboxes, 0.5);
        int &selectedIdx = nmsResults.first;
        std::vector<float> &bbox = nmsResults.second;
        // // parameters, translations
        std::vector<float> outputCoord = outputCoords[selectedIdx];
        // outputCoord[0] = outputCoords[selectedIdx];
        // std::cout << outputCoords.size() << std::endl;
        std::pair<std::vector<float>, std::vector<float>> ParamsKps = GetParamsKpsWithConv(i, x, outputCoord, xShape);
        std::vector<float> &param = ParamsKps.first;
        std::vector<float> &tran = ParamsKps.second;
        // std::vector<float> &param = ParamsKps.first[selectedIdx];
        // std::vector<float> &tran = ParamsKps.second[selectedIdx];
        std::vector<std::vector<float>> landmarks = GetLandmarks(param);
        ResizedBboxLandmarks(resizedRatios, bbox, landmarks, tran);
        results.first.push_back(bbox);
        results.second.push_back(landmarks);
        // for (int i = 0; i < 68; i++)
        // {
        //     std::cout << "(" << landmarks[i][0] << ", " << landmarks[i][1] << ")" << std::endl;
        // }
        // return std::make_pair(bbox, landmarks);
    }
    std::vector<std::vector<float>> PostProcessor::FindCoord(float *cls_x,
                                                             const std::vector<int> &cls_xShape,
                                                             bool needSigmoid)
    {
        float i, j;
        int height = cls_xShape[2], width = cls_xShape[2], channel = cls_xShape[3];
        int o_pos;
        float val0, val1, val2, val3;
        std::vector<std::vector<float>> outputCoords;
        outputCoords.reserve(15);
        for (j = 0; j < height; j++)
        {
            for (i = 0; i < width; i++)
            {
                o_pos = j * (width * channel) + i * channel;
                val0 = *(cls_x + o_pos);
                if (needSigmoid)
                    val0 = 1 / (1 + exp(-val0));
                if (val0 > this->thresholdVal)
                {
                    std::vector<float> coord{j, i, 0, val0};
                    outputCoords.emplace_back(coord);
                }
                o_pos++;
                val1 = *(cls_x + o_pos);
                if (needSigmoid)
                    val1 = 1 / (1 + exp(-val1));
                if (val1 > this->thresholdVal)
                {
                    std::vector<float> coord{j, i, 1, val1};
                    outputCoords.emplace_back(coord);
                }
                o_pos++;
                val2 = *(cls_x + o_pos);
                if (needSigmoid)
                    val2 = 1 / (1 + exp(-val2));
                if (val2 > this->thresholdVal)
                {
                    std::vector<float> coord{j, i, 2, val2};
                    outputCoords.emplace_back(coord);
                }
                o_pos++;
                val3 = *(cls_x + o_pos);
                if (needSigmoid)
                    val3 = 1 / (1 + exp(-val3));
                if (val3 > this->thresholdVal)
                {
                    std::vector<float> coord{j, i, 3, val3};
                    outputCoords.emplace_back(coord);
                }
            }
        }
        return outputCoords;
    }

    std::vector<std::vector<float>> PostProcessor::GetBboxWithConv(int stride_idx,
                                                                   float *bbox_x,
                                                                   std::vector<std::vector<float>> &coords,
                                                                   const std::vector<int> &bbox_xShape)
    {
        std::vector<std::vector<float>> outputBbox;
        outputBbox.reserve(15);
        int input_channel = 64,
            output_channel = 8,
            stride_width = 40,
            stride = 8;
        int meshx, meshy;
        int loc_y, loc_x, feat_pos, weight_pos;
        int detSize = coords.size();
        std::vector<int> meshX = meshX40, meshY = meshY40;
        if (stride_idx == 1)
        {
            stride_width = 20;
            meshX = meshX20;
            meshY = meshY20;
            stride = strideFPN[stride_idx];
        }

        else if (stride_idx == 2)
        {
            stride_width = 10;
            meshX = meshX10;
            meshY = meshY10;
            stride = strideFPN[stride_idx];
        }

        int map_width = stride_width * input_channel, kernel_size = 3;
        int c;
        int l, i, j, k_in, k_out, end;
        float score;
        std::vector<float> coord;
        std::string stride_index = std::to_string(stride_idx);
        // (3, 3, 8, 64)
        float *weight_data = std::get<1>(headWeights["weight_pred_bbox_" + stride_index]);
        float *bias_data = std::get<1>(headWeights["bias_pred_bbox_" + stride_index]);
        float *scale_data = std::get<1>(headWeights["scale_bbox_" + stride_index + ":0"]);
        float val = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
        for (l = 0; l < detSize; l++)
        {
            coord = coords[l];
            c = coord[2] / 4;
            score = coord[3];
            std::vector<float> bbox;
            meshx = meshX[coord[1]] * stride;
            meshy = meshY[coord[0]] * stride;
            bbox.reserve(6);
            if (c == 0)
            {
                k_out = 0;
                end = 4;
            }
            else
            {
                k_out = 4;
                end = 8;
            }
            for (; k_out < end; k_out++)
            {
                val = 0.0f, v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                for (j = 0; j < kernel_size; j++)
                {
                    for (i = 0; i < kernel_size; i++)
                    {
                        for (k_in = 0; k_in < input_channel; k_in += 4)
                        {
                            loc_y = grid3x3[i][j][1] + coord[0];
                            loc_x = grid3x3[i][j][0] + coord[1];
                            // For stride 4
                            feat_pos = loc_y * (map_width) + loc_x * input_channel + k_in;
                            weight_pos = j * kernel_size * output_channel * input_channel + i * output_channel * input_channel + k_out * input_channel + k_in;
                            v0 = *(bbox_x + feat_pos) * (*(weight_data + weight_pos));
                            feat_pos++;
                            weight_pos++;
                            v1 = *(bbox_x + feat_pos) * (*(weight_data + weight_pos));
                            feat_pos++;
                            weight_pos++;
                            v2 = *(bbox_x + feat_pos) * (*(weight_data + weight_pos));
                            feat_pos++;
                            weight_pos++;
                            v3 = *(bbox_x + feat_pos) * (*(weight_data + weight_pos));
                            val = val + v0 + v1 + v2 + v3;
                        }
                    }
                }
                val = (val + *(bias_data + k_out)) * (*scale_data);
                val *= stride;
                bbox.emplace_back(val);
            }
            bbox.emplace_back(score);
            bbox.emplace_back(c);
            bbox[0] = MIN(MAX(0, meshx - bbox[0]), stride_width * stride);
            bbox[1] = MIN(MAX(0, meshy - bbox[1]), stride_width * stride);
            bbox[2] = MIN(MAX(0, meshx + bbox[2]), stride_width * stride);
            bbox[3] = MIN(MAX(0, meshy + bbox[3]), stride_width * stride);
            outputBbox.emplace_back(bbox);
        }

        return outputBbox;
    }
    std::pair<std::vector<float>, std::vector<float>> PostProcessor::GetParamsKpsWithConv(int stride_idx,
                                                                                          float *x,
                                                                                          std::vector<float> &coord,
                                                                                          const std::vector<int> &xShape)
    {
        std::vector<std::vector<float>> outputParams, outputKps;
        outputParams.reserve(15);
        outputKps.reserve(15);
        // two conv3x3
        // one pred conv1x1
        // conv3x3 (32, 64) ->(64, 64)-> conv1x1()
        int meshx, meshy;
        std::vector<int> meshX = meshX40, meshY = meshY40;
        int stride_width = 40,
            stride = stride = strideFPN[stride_idx];
        if (stride_idx == 1)
        {
            stride_width = 20;
            meshX = meshX20;
            meshY = meshY20;
        }
        else if (stride_idx == 2)
        {
            stride_width = 10;
            meshX = meshX10;
            meshY = meshY10;
        }
        int input_channel1 = 32,
            output_channel1 = 64;
        int input_channel2 = 64,
            output_channel2 = 64;
        int input_pred_channels = 64,
            output_param_pred_channels = 120,
            output_kp_pred_channels = 4;
        int map_width1 = stride_width * input_channel1, kernel_size = 3;
        int map_width2 = stride_width * input_channel2;
        int c;
        int l, i, j, k_in, k_out, ks_i, ks_j, end;
        int loc_y, loc_x, feat_pos, weight_pos;
        int buffer1 = 3 * 3 * 64 * sizeof(float), buffer2 = 64 * sizeof(float);
        float param_val = 0.0f, param_v0 = 0.0f, param_v1 = 0.0f, param_v2 = 0.0f, param_v3 = 0.0f;
        float param_conv3x3_output_1[3][3][64] = {}, param_conv3x3_output_2[64] = {};
        float kp_conv3x3_output_1[3][3][64] = {}, kp_conv3x3_output_2[64] = {};
        float kp_val = 0.0f, kp_v0 = 0.0f, kp_v1 = 0.0f, kp_v2 = 0.0f, kp_v3 = 0.0f;
        // std::vector<float> coord;
        std::string stride_index = std::to_string(stride_idx);
        // (3, 3, 64, 32)
        float *param_weight_conv3x3_data1 = std::get<1>(headWeights["weight_conv3x3_params_" + stride_index + "_0"]);
        float *param_bias_conv3x3_data1 = std::get<1>(headWeights["bias_conv3x3_params_" + stride_index + "_0"]);
        float *param_weight_conv3x3_data2 = std::get<1>(headWeights["weight_conv3x3_params_" + stride_index + "_1"]);
        float *param_bias_conv3x3_data2 = std::get<1>(headWeights["bias_conv3x3_params_" + stride_index + "_1"]);
        float *param_weight_pred_data = std::get<1>(headWeights["weight_pred_params_" + stride_index]);
        float *param_bias_pred_data = std::get<1>(headWeights["bias_pred_params_" + stride_index]);

        float *kp_weight_conv3x3_data1 = std::get<1>(headWeights["weight_conv3x3_kps_" + stride_index + "_0"]);
        float *kp_bias_conv3x3_data1 = std::get<1>(headWeights["bias_conv3x3_kps_" + stride_index + "_0"]);
        float *kp_weight_conv3x3_data2 = std::get<1>(headWeights["weight_conv3x3_kps_" + stride_index + "_1"]);
        float *kp_bias_conv3x3_data2 = std::get<1>(headWeights["bias_conv3x3_kps_" + stride_index + "_1"]);
        float *kp_weight_pred_data = std::get<1>(headWeights["weight_pred_kp_" + stride_index]);
        float *kp_bias_pred_data = std::get<1>(headWeights["bias_pred_kp_" + stride_index]);
        // for (l = 0; l < coords.size(); l++)
        // {
        // coord = coords[l];
        c = coord[2] / 4;
        meshx = meshX[coord[1]] * stride;
        meshy = meshY[coord[0]] * stride;
        memset(param_conv3x3_output_1, 0.0f, buffer1);
        memset(kp_conv3x3_output_1, 0.0f, buffer1);
        //==================================================================
        // for first conv3x3
        for (ks_j = 0; ks_j < kernel_size; ks_j++)
        {
            for (ks_i = 0; ks_i < kernel_size; ks_i++)
            {
                for (k_out = 0; k_out < output_channel1; k_out++)
                {
                    param_val = 0.0, param_v0 = 0.0, param_v1 = 0.0, param_v2 = 0.0, param_v3 = 0.0;
                    kp_val = 0.0, kp_v0 = 0.0, kp_v1 = 0.0, kp_v2 = 0.0, kp_v3 = 0.0;
                    for (j = ks_j; j < ks_j + kernel_size; j++)
                    {
                        for (i = ks_i; i < ks_i + kernel_size; i++)
                        {
                            for (k_in = 0; k_in < input_channel1; k_in += 4)
                            {
                                loc_y = grid5x5[i][j][1] + coord[0];
                                loc_x = grid5x5[i][j][0] + coord[1];
                                // For stride 4
                                feat_pos = loc_y * (map_width1) + loc_x * input_channel1 + k_in;
                                weight_pos = (j - ks_j) * kernel_size * output_channel1 * input_channel1 +
                                             (i - ks_i) * output_channel1 * input_channel1 + k_out * input_channel1 + k_in;
                                param_v0 = *(x + feat_pos) * (*(param_weight_conv3x3_data1 + weight_pos));
                                kp_v0 = *(x + feat_pos) * (*(kp_weight_conv3x3_data1 + weight_pos));

                                feat_pos++;
                                weight_pos++;
                                param_v1 = *(x + feat_pos) * (*(param_weight_conv3x3_data1 + weight_pos));
                                kp_v1 = *(x + feat_pos) * (*(kp_weight_conv3x3_data1 + weight_pos));

                                feat_pos++;
                                weight_pos++;
                                param_v2 = *(x + feat_pos) * (*(param_weight_conv3x3_data1 + weight_pos));
                                kp_v2 = *(x + feat_pos) * (*(kp_weight_conv3x3_data1 + weight_pos));

                                feat_pos++;
                                weight_pos++;
                                param_v3 = *(x + feat_pos) * (*(param_weight_conv3x3_data1 + weight_pos));
                                kp_v3 = *(x + feat_pos) * (*(kp_weight_conv3x3_data1 + weight_pos));
                                param_val = param_val + param_v0 + param_v1 + param_v2 + param_v3;
                                kp_val += kp_v0 + kp_v1 + kp_v2 + kp_v3;
                            }
                        }
                    }

                    param_val = (param_val + *(param_bias_conv3x3_data1 + k_out));
                    kp_val += (*(kp_bias_conv3x3_data1 + k_out));
                    // std::cout << val << std::endl;
                    param_conv3x3_output_1[ks_j][ks_i][k_out] = MAX(0, param_val);
                    kp_conv3x3_output_1[ks_j][ks_i][k_out] = MAX(0, kp_val);
                }
            }
        }
        //==================================================================
        // for second conv3x3
        memset(param_conv3x3_output_2, 0.0f, buffer2);
        memset(kp_conv3x3_output_2, 0.0f, buffer2);
        for (k_out = 0; k_out < output_channel2; k_out++)
        {
            param_val = 0.0f, param_v0 = 0.0, param_v1 = 0.0f, param_v2 = 0.0f, param_v3 = 0.0f;
            kp_val = 0.0f, kp_v0 = 0.0f, kp_v1 = 0.0f, kp_v2 = 0.0f, kp_v3 = 0.0f;
            // printf("----------------------------\n");
            for (j = 0; j < kernel_size; j++)
            {
                for (i = 0; i < kernel_size; i++)
                {
                    for (k_in = 0; k_in < input_channel2; k_in += 4)
                    {
                        // For stride 4
                        feat_pos = k_in;
                        weight_pos = j * kernel_size * output_channel2 * input_channel2 +
                                     i * output_channel2 * input_channel2 + k_out * input_channel2 + k_in;
                        param_v0 = param_conv3x3_output_1[j][i][feat_pos] * (*(param_weight_conv3x3_data2 + weight_pos));
                        kp_v0 = kp_conv3x3_output_1[j][i][feat_pos] * (*(kp_weight_conv3x3_data2 + weight_pos));

                        feat_pos++;
                        weight_pos++;
                        param_v1 = param_conv3x3_output_1[j][i][feat_pos] * (*(param_weight_conv3x3_data2 + weight_pos));
                        kp_v1 = kp_conv3x3_output_1[j][i][feat_pos] * (*(kp_weight_conv3x3_data2 + weight_pos));

                        feat_pos++;
                        weight_pos++;

                        param_v2 = param_conv3x3_output_1[j][i][feat_pos] * (*(param_weight_conv3x3_data2 + weight_pos));
                        kp_v2 = kp_conv3x3_output_1[j][i][feat_pos] * (*(kp_weight_conv3x3_data2 + weight_pos));

                        feat_pos++;
                        weight_pos++;

                        param_v3 = param_conv3x3_output_1[j][i][feat_pos] * (*(param_weight_conv3x3_data2 + weight_pos));
                        kp_v3 = kp_conv3x3_output_1[j][i][feat_pos] * (*(kp_weight_conv3x3_data2 + weight_pos));

                        param_val = param_val + param_v0 + param_v1 + param_v2 + param_v3;
                        kp_val += kp_v0 + kp_v1 + kp_v2 + kp_v3;
                    }
                }
            }
            param_val = (param_val + *(param_bias_conv3x3_data2 + k_out));
            kp_val += (*(kp_bias_conv3x3_data2 + k_out));
            param_conv3x3_output_2[k_out] = MAX(0, param_val);
            kp_conv3x3_output_2[k_out] = MAX(0, kp_val);
        }
        std::vector<float> params(output_param_pred_channels / 2, 0);
        if (c == 0)
        {
            k_out = 0;
            end = output_param_pred_channels / 2;
        }
        else
        {
            k_out = output_param_pred_channels / 2;
            end = output_param_pred_channels;
        }
        // printf("----------------------------\n");
        for (; k_out < end; k_out++)
        {
            param_val = 0.0, param_v0 = 0.0, param_v1 = 0.0, param_v2 = 0.0, param_v3 = 0.0;
            for (k_in = 0; k_in < input_pred_channels; k_in = (k_in + 4))
            {
                feat_pos = k_in;
                weight_pos = k_out * input_pred_channels + k_in;
                param_v0 = param_conv3x3_output_2[feat_pos] * (*(param_weight_pred_data + weight_pos));

                feat_pos++;
                weight_pos++;
                param_v1 = param_conv3x3_output_2[feat_pos] * (*(param_weight_pred_data + weight_pos));

                feat_pos++;
                weight_pos++;
                param_v2 = param_conv3x3_output_2[feat_pos] * (*(param_weight_pred_data + weight_pos));

                feat_pos++;
                weight_pos++;
                param_v3 = param_conv3x3_output_2[feat_pos] * (*(param_weight_pred_data + weight_pos));
                param_val += param_v0 + param_v1 + param_v2 + param_v3;
            }
            param_val += (*(param_bias_pred_data + k_out));
            // std::cout << val << std::endl;
            params[k_out] = param_val;
        }
        // outputParams.emplace_back(params);
        // printf("----------------------------\n");
        std::vector<float> kps(output_kp_pred_channels / 2, 0);
        // std::cout << output_kp_pred_channels << std::endl;
        if (c == 0)
        {
            k_out = 0;
            end = output_kp_pred_channels / 2;
        }
        else
        {
            k_out = 2;
            end = output_kp_pred_channels;
        }

        for (; k_out < end; k_out++)
        {
            kp_val = 0.0f, kp_v0 = 0.0f, kp_v1 = 0.0f, kp_v2 = 0.0f, kp_v3 = 0.0f;
            for (k_in = 0; k_in < input_pred_channels; k_in = (k_in + 4))
            {
                feat_pos = k_in;
                weight_pos = k_out * input_pred_channels + k_in;
                kp_v0 = kp_conv3x3_output_2[feat_pos] * (*(kp_weight_pred_data + weight_pos));

                feat_pos++;
                weight_pos++;
                kp_v1 = kp_conv3x3_output_2[feat_pos] * (*(kp_weight_pred_data + weight_pos));

                feat_pos++;
                weight_pos++;
                kp_v2 = kp_conv3x3_output_2[feat_pos] * (*(kp_weight_pred_data + weight_pos));

                feat_pos++;
                weight_pos++;
                kp_v3 = kp_conv3x3_output_2[feat_pos] * (*(kp_weight_pred_data + weight_pos));
                kp_val += kp_v0 + kp_v1 + kp_v2 + kp_v3;
            }
            kp_val = (kp_val + *(kp_bias_pred_data + k_out)) * stride;
            // std::cout << kp_val << std::endl;
            kps[k_out] = kp_val;
        }
        kps[0] = MIN(MAX(0, meshx + kps[0]), stride_width * stride);
        kps[1] = MIN(MAX(0, meshy + kps[1]), stride_width * stride);
        // std::cout << kps[0] << std::endl;
        // std::cout << kps[1] << std::endl;
        // bbox[2] = MIN(MAX(0, meshx + bbox[2]), stride_width * stride);
        // bbox[3] = MIN(MAX(0, meshy + bbox[3]), stride_width * stride);
        // outputKps.emplace_back(kps);
        // }
        // exit(1);
        return std::make_pair(params, kps);
    }

    std::vector<std::vector<float>> PostProcessor::GetBbox(int stride_idx,
                                                           float *bbox_x,
                                                           std::vector<std::vector<float>> &coords,
                                                           const std::vector<int> &bbox_xShape)
    {
        std::vector<std::vector<float>> outputBbox;
        outputBbox.reserve(15);
        int output_channel = 8,
            stride_width = bbox_xShape[1],
            stride = 8;
        int meshx, meshy, loc_y, loc_x, feat_pos;
        int detSize = coords.size();
        std::vector<int> meshX = meshX40, meshY = meshY40;
        if (stride_idx == 1)
        {
            stride_width = 20;
            meshX = meshX20;
            meshY = meshY20;
            stride = strideFPN[stride_idx];
        }

        else if (stride_idx == 2)
        {
            stride_width = 10;
            meshX = meshX10;
            meshY = meshY10;
            stride = strideFPN[stride_idx];
        }
        int l, i, j, c, k_out, end;
        float score, val;
        std::vector<float> coord;
        int map_width = stride_width * output_channel;
        for (l = 0; l < detSize; l++)
        {
            coord = coords[l];
            c = coord[2] / 4;
            score = coord[3];
            std::vector<float> bbox(6, 0);
            meshx = meshX[coord[1]] * stride;
            meshy = meshY[coord[0]] * stride;
            // bbox.reserve(6);
            if (c == 0)
            {
                k_out = 0;
            }
            else
            {
                k_out = 4;
            }
            loc_y = coord[0];
            loc_x = coord[1];
            feat_pos = loc_y * map_width + loc_x * output_channel + k_out;
            val = *(bbox_x + feat_pos) * stride;

            bbox[0] = MIN(MAX(0, meshx - val), stride_width * stride);
            feat_pos++;
            val = *(bbox_x + feat_pos) * stride;

            bbox[1] = MIN(MAX(0, meshy - val), stride_width * stride);
            // std::cout << meshy - val << std::endl;
            feat_pos++;
            val = *(bbox_x + feat_pos) * stride;
            bbox[2] = MIN(MAX(0, meshx + val), stride_width * stride);
            feat_pos++;
            val = *(bbox_x + feat_pos) * stride;
            bbox[3] = MIN(MAX(0, meshy + val), stride_width * stride);
            bbox[4] = score;
            bbox[5] = c;
            outputBbox.emplace_back(bbox);
        }
        // std::cout << outputBbox[0][0] << std::endl;
        // std::cout << outputBbox[0][1] << std::endl;
        // std::cout << outputBbox[0][2] << std::endl;
        // std::cout << outputBbox[0][3] << std::endl;
        // exit(1);
        return outputBbox;
    }

    std::pair<std::vector<float>, std::vector<float>> PostProcessor::GetParamsKps(int stride_idx,
                                                                                  std::vector<float> &coord,
                                                                                  float *param_x,
                                                                                  float *trans_x)
    {
        int meshx, meshy;
        std::vector<int> meshX = meshX40, meshY = meshY40;
        int stride_width = 40,
            stride = stride = strideFPN[stride_idx];
        if (stride_idx == 1)
        {
            stride_width = 20;
            meshX = meshX20;
            meshY = meshY20;
        }
        else if (stride_idx == 2)
        {
            stride_width = 10;
            meshX = meshX10;
            meshY = meshY10;
        }

        int output_param_pred_channels = 120, output_kp_pred_channels = 4;
        int param_map_width = stride_width * output_param_pred_channels;
        int kp_map_width = stride_width * output_kp_pred_channels;
        int c;
        int l, i, j, k_out, end;
        int loc_y, loc_x, feat_pos;
        c = coord[2] / 4;
        meshx = meshX[coord[1]] * stride;
        meshy = meshY[coord[0]] * stride;
        std::vector<float> params(output_param_pred_channels / 2, 0);
        loc_y = coord[0];
        loc_x = coord[1];
        if (c == 0)
        {
            k_out = 0;
            end = output_param_pred_channels / 2;
        }
        else
        {
            k_out = output_param_pred_channels / 2;
            end = output_param_pred_channels;
        }
        for (i = 0; k_out < end; k_out++, i++)
        {
            // printf("--------------\n");
            feat_pos = loc_y * param_map_width + loc_x * output_param_pred_channels + k_out;
            params[i] = *(param_x + feat_pos);
            // std::cout << params[k_out] << std::endl;
        }
        std::vector<float> kps(output_kp_pred_channels / 2, 0);
        // std::cout << output_kp_pred_channels << std::endl;
        if (c == 0)
        {
            k_out = 0;
            end = output_kp_pred_channels / 2;
        }
        else
        {
            k_out = 2;
            end = output_kp_pred_channels;
        }

        for (i = 0; k_out < end; k_out++, i++)
        {
            feat_pos = loc_y * kp_map_width + loc_x * output_kp_pred_channels + k_out;
            kps[i] = *(trans_x + feat_pos) * stride;
        }
        kps[0] = MIN(MAX(0, meshx + kps[0]), stride_width * stride);
        kps[1] = MIN(MAX(0, meshy + kps[1]), stride_width * stride);
        return std::make_pair(params, kps);
    }
    std::vector<std::vector<float>> PostProcessor::GetLandmarks(std::vector<float> &param)
    {
        int R = 9, shp = 40, exp = 11;
        int numParams = R + shp + exp;
        int i, j, pos, fill_pos;
        float *pm = std::get<1>(BFM["pms"]);
        float *ps = std::get<1>(BFM["pms"]) + numParams;
        float *u_base = std::get<1>(BFM["u_base"]);
        float *shp_base = std::get<1>(BFM["shp_base"]);
        float *exp_base = std::get<1>(BFM["exp_base"]);
        float val;
        std::vector<std::vector<float>> outputLandmarks(69, std::vector<float>(2, 0));
        float lnmks[204] = {};
        for (i = 0; i < numParams; i = (i + 2))
        {
            pos = i;
            param[pos] = param[pos] * (*(ps + pos)) + (*(pm + pos));
            pos++;
            param[pos] = param[pos] * (*(ps + pos)) + (*(pm + pos));
        }
        for (i = 0; i < 204; i++)
        {
            lnmks[i] = *(u_base + i);

            val = 0.0;
            for (j = 0; j < shp; j++)
            {
                pos = i * shp + j;
                val += *(shp_base + pos) * param[j + R];
            }
            lnmks[i] += val;
            val = 0.0;
            for (j = 0; j < exp; j++)
            {
                pos = i * exp + j;
                val += *(exp_base + pos) * param[j + R + shp];
            }
            lnmks[i] += val;
        }
        float min_lnmk_w = 0.0f, max_lnmk_w = 0.0f;
        float min_lnmk_h = 0.0f, max_lnmk_h = 0.0f;
        for (i = 0, j = 0; i < 68; i++)
        {
            val = 0.0;
            pos = i * 3;
            val += lnmks[pos] * param[0];
            pos++;
            val += lnmks[pos] * param[1];
            pos++;
            val += lnmks[pos] * param[2];
            outputLandmarks[i][0] = val;
            // min_lnmk_w = val;
            if (min_lnmk_w > val)
                min_lnmk_w = val;
            if (max_lnmk_w < val)
                max_lnmk_w = val;
            pos = i * 3;
            val = 0.0;
            val += lnmks[pos] * param[3];
            pos++;
            val += lnmks[pos] * param[4];
            pos++;
            val += lnmks[pos] * param[5];
            outputLandmarks[i][1] = val;
            if (min_lnmk_h > val)
                min_lnmk_h = val;
            if (max_lnmk_h < val)
                max_lnmk_h = val;
            // pos = i * 3;
            // val = 0.0;
            // val += lnmks[pos] * param[6];
            // pos++;
            // val += lnmks[pos] * param[7];
            // pos++;
            // val += lnmks[pos] * param[8];
            // outputLandmarks[i][2] = val;
        }
        outputLandmarks[68][0] = (max_lnmk_w - min_lnmk_w);
        outputLandmarks[68][1] = (max_lnmk_h - min_lnmk_h);
        return outputLandmarks;
    }
    void PostProcessor::ResizedBboxLandmarks(float *resizedRatios, std::vector<float> &bbox, std::vector<std::vector<float>> &landmarks, std::vector<float> &tran)
    {
        int i, j, pos;
        bbox[0] *= resizedRatios[0];
        bbox[1] *= resizedRatios[1];
        bbox[2] *= resizedRatios[0];
        bbox[3] *= resizedRatios[1];
        tran[0] *= resizedRatios[0];
        tran[0] += 0.5f;
        tran[1] *= resizedRatios[1];
        tran[1] += 0.5f;
        float objw = (bbox[2] - bbox[0]);
        float objh = (bbox[3] - bbox[1]);
        float lnmkw = landmarks[68][0];
        float lnmkh = landmarks[68][1];
        float lnmk2bbox_w_ratio = objw / lnmkw;
        float lnmk2bbox_h_ratio = objh / lnmkh;
        for (i = 0; i < landmarks.size() - 1; i = (i + 2))
        {
            pos = i;
            landmarks[pos][0] = landmarks[pos][0] * lnmk2bbox_w_ratio + tran[0];
            landmarks[pos][1] = landmarks[pos][1] * lnmk2bbox_h_ratio + tran[1];
            pos++;
            landmarks[pos][0] = landmarks[pos][0] * lnmk2bbox_w_ratio + tran[0];
            landmarks[pos][1] = landmarks[pos][1] * lnmk2bbox_h_ratio + tran[1];
        }
    }
}
