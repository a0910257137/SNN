#include "Interpreter.h"
// #include <inttypes.h>
namespace SNN
{
    Interpreter::Interpreter(string model_path)
    {
        threads = -1;
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        TFLITE_MINIMAL_CHECK(model != nullptr);
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        unique_ptr<tflite::Interpreter> tflite_interpreter(new tflite::Interpreter());
        builder.PreserveAllTensorsExperimental();
        builder(&this->tflite_interpreter, threads);
        TFLITE_MINIMAL_CHECK((this->tflite_interpreter) != nullptr);
    }
    Interpreter::~Interpreter() {}

    void Interpreter::IdentifyOperation(shared_ptr<Tensor> tensor, std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory, const TfLiteNode &node, tflite::BuiltinOperator tflite_op)
    {

        TfLiteTensor *tflite_tensor;
        int i, j, ptr_size;
        pair<float *, float *> weight_bias;
        int dimSizes, weightBytes, biasBytes;
        if (tflite_op == tflite::BuiltinOperator_DEPTHWISE_CONV_2D)
        {
            TfLiteDepthwiseConvParams *tflite_params = (TfLiteDepthwiseConvParams *)node.builtin_data;
            tensor->SetStride(0, tflite_params->stride_height);
            tensor->SetStride(1, tflite_params->stride_width);
            tensor->SetDilation(0, tflite_params->dilation_height_factor);
            tensor->SetDilation(1, tflite_params->dilation_width_factor);
            tensor->SetPaddingType(tflite_params->padding);
            tensor->SetOpType(DepthwiseConv);
            tensor->SetActType(tflite_params->activation);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            for (j = 0; j < 4; ++j)
            {
                tensor->SetInputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {
                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
            ptr_size = (node.inputs->size - 1) / 2;

            for (i = 0; i < ptr_size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 1]);
                dimSizes = tflite_tensor->dims->data[0] * tflite_tensor->dims->data[1] * tflite_tensor->dims->data[2] * tflite_tensor->dims->data[3];
                weightBytes = dimSizes * sizeof(float);
                tensor->SetWeightBytes(static_cast<uint32_t>(weightBytes));
                float *src = tflite_tensor->data.f;
                weight_bias.first = (float *)malloc(weightBytes);
                // MHWI->MIHW
                Tranpose(src, weight_bias.first, FILTER_FORMAT_MHWI, FILTER_FORMAT_MIHW, tflite_tensor->dims->data);
                // for (int i = 0; i < 27; i++)
                // {
                //     std::cout << src[i] << std::endl;
                // }
                // exit(1);
                // int buffer_sizes = 1 * 3 * 3 * 3 * sizeof(float);
                // float *ws = (float *)malloc(buffer_sizes);
                // FILE *ptr;
                // const char *char_path = "/aidata/anders/data_collection/okay/WF/archives/test/test_data/w.bin";
                // ptr = fopen(char_path, "rb");
                // fread(ws, buffer_sizes, 1, ptr);
                // for (int i = 0; i < 27; i++)
                // {

                //     printf("%f\n", ws[i]);
                // }
                // exit(1);
                // weight_bias.first = ws;
                tensor->SetKernelShape(0, tflite_tensor->dims->data[0]);
                tensor->SetKernelShape(1, tflite_tensor->dims->data[3]);
                tensor->SetKernelShape(2, tflite_tensor->dims->data[1]);
                tensor->SetKernelShape(3, tflite_tensor->dims->data[2]);

                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 2]);
                tensor->SetBiasBytes(static_cast<uint32_t>(weightBytes));
                dimSizes = tflite_tensor->dims->data[0];
                biasBytes = dimSizes * sizeof(float);
                weight_bias.second = (float *)malloc(biasBytes);
                memcpy(weight_bias.second, tflite_tensor->data.f, biasBytes);
                tensor->SetBiasShape(0, tflite_tensor->dims->data[0]);
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_CONV_2D)
        {

            TfLiteConvParams *tflite_params = (TfLiteConvParams *)node.builtin_data;
            tensor->SetStride(0, tflite_params->stride_height);
            tensor->SetStride(1, tflite_params->stride_width);
            tensor->SetDilation(0, tflite_params->dilation_height_factor);
            tensor->SetDilation(1, tflite_params->dilation_width_factor);
            tensor->SetPaddingType(tflite_params->padding);
            tensor->SetOpType(Conv2D);
            tensor->SetActType(tflite_params->activation);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            for (j = 0; j < 4; ++j)
            {
                tensor->SetInputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {
                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
            ptr_size = (node.inputs->size - 1) / 2;

            for (i = 0; i < ptr_size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 1]);
                dimSizes = tflite_tensor->dims->data[0] * tflite_tensor->dims->data[1] * tflite_tensor->dims->data[2] * tflite_tensor->dims->data[3];
                weightBytes = dimSizes * sizeof(float);
                tensor->SetWeightBytes(static_cast<uint32_t>(weightBytes));
                float *src = tflite_tensor->data.f;
                weight_bias.first = (float *)malloc(weightBytes);
                Tranpose(src, weight_bias.first, FILTER_FORMAT_OHWI, FILTER_FORMAT_OIHW, tflite_tensor->dims->data);
                // exit(1);
                tensor->SetKernelShape(0, tflite_tensor->dims->data[0]);
                tensor->SetKernelShape(1, tflite_tensor->dims->data[3]);
                tensor->SetKernelShape(2, tflite_tensor->dims->data[1]);
                tensor->SetKernelShape(3, tflite_tensor->dims->data[2]);
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 2]);
                dimSizes = tflite_tensor->dims->data[0];
                biasBytes = dimSizes * sizeof(float);
                tensor->SetBiasBytes(static_cast<uint32_t>(biasBytes));
                weight_bias.second = (float *)malloc(biasBytes);
                memcpy(weight_bias.second, tflite_tensor->data.f, biasBytes);
                tensor->SetBiasShape(0, tflite_tensor->dims->data[0]);
            }
        }
        tensor->SetMemoryPtr(ptr_size);
        int start_index = mainMemory->size();
        for (j = start_index; j < start_index + ptr_size; j++)
        {
            tensor->SetMemoryPtrIndex(j - start_index, static_cast<uint8_t>(start_index));
        }
        mainMemory->push_back(weight_bias);
    }
    vector<shared_ptr<Tensor>> Interpreter::mGraphToSNNGraph(std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory)
    {
        int node_index, i, j;
        vector<int> execution_plan = this->tflite_interpreter->execution_plan();
        vector<shared_ptr<Tensor>> GraphNodes;
        tflite::BuiltinOperator tflite_op;
        printf("INFO: Start converting Tf-Lite graph to SNN graph ... \n");
        for (i = 0; i < execution_plan.size(); ++i)
        {
            node_index = execution_plan[i];
            shared_ptr<Tensor> tensor(new Tensor()); // default tensor
            const pair<TfLiteNode, TfLiteRegistration> *node_and_registration = (this->tflite_interpreter)->node_and_registration(node_index);
            const TfLiteNode &node = node_and_registration->first;
            const TfLiteRegistration &registration = node_and_registration->second;
            tflite_op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
            string op_name = tflite::GetOpNameByRegistration(registration);
            // printf("%s\n", op_name.c_str());
            for (j = 0; j < node.inputs->size; j++)
                tensor->inputIndex.push_back(node.inputs->data[i]);
            for (j = 0; j < node.outputs->size; j++)
                tensor->outputIndex.push_back(node.outputs->data[i]);
            IdentifyOperation(tensor, mainMemory, node, tflite_op);
            GraphNodes.push_back(tensor);
        }
        return GraphNodes;
    }
    void Interpreter::Tranpose(float *src, float *dst, FilterFormat inFormat, FilterFormat outFormat, int *shapDims)
    {

        if (inFormat == FILTER_FORMAT_OHWI && outFormat == FILTER_FORMAT_OIHW)
        {
            int i, o, h, w;
            int O = shapDims[0], H = shapDims[1], W = shapDims[2], I = shapDims[3];
            for (o = 0; o < O; ++o)
            {
                for (i = 0; i < I; ++i)
                {
                    for (h = 0; h < H; ++h)
                    {
                        for (w = 0; w < W; ++w)
                        {

                            int fromIndex = o * H * W * I + h * W * I + w * I + i;
                            int toIndex = o * I * H * W + i * H * W + h * W + w;
                            dst[toIndex] = src[fromIndex];
                            // printf("%f ", dst[toIndex]);
                        }
                        // printf("\n");
                    }
                    // printf("\n");
                }
                // printf("\n");
            }
        }
        else if (inFormat == FILTER_FORMAT_MHWI && outFormat == FILTER_FORMAT_MIHW)
        {
            int m, i, h, w;
            int M = shapDims[0], H = shapDims[1], W = shapDims[2], I = shapDims[3];
            for (m = 0; m < M; ++m)
            {
                for (i = 0; i < I; ++i)
                {
                    for (h = 0; h < H; ++h)
                    {
                        for (w = 0; w < W; ++w)
                        {

                            int fromIndex = m * H * W * I + h * W * I + w * I + i;
                            int toIndex = m * I * H * W + i * H * W + h * W + w;
                            dst[toIndex] = src[fromIndex];
                            // printf("%f ", dst[toIndex]);
                        }
                        // printf("\n");
                    }
                    // printf("\n");
                }
                // printf("\n");
            }
        }
    }
}