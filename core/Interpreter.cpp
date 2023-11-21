#include "Interpreter.h"
// #include <inttypes.h>
namespace SNN
{
    Interpreter::Interpreter(std::string model_path)
    {
        threads = -1;
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        TFLITE_MINIMAL_CHECK(model != nullptr);
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        std::unique_ptr<tflite::Interpreter> tflite_interpreter(new tflite::Interpreter());
        builder.PreserveAllTensorsExperimental();
        builder(&this->tflite_interpreter, threads);
        TFLITE_MINIMAL_CHECK((this->tflite_interpreter) != nullptr);
    }
    Interpreter::~Interpreter()
    {
    }

    void Interpreter::IdentifyOperation(shared_ptr<Tensor> tensor,
                                        std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory,
                                        const TfLiteNode &node,
                                        tflite::BuiltinOperator tflite_op)
    {

        TfLiteTensor *tflite_tensor;
        int i, j, ptr_size;
        std::pair<float *, float *> weight_bias;
        int dimSizes, weightBytes, biasBytes;
        tensor->SetType(DataType_DT_FLOAT);
        if (tflite_op == tflite::BuiltinOperator_DEPTHWISE_CONV_2D)
        {
            TfLiteDepthwiseConvParams *tflite_params = (TfLiteDepthwiseConvParams *)node.builtin_data;
            tensor->SetStride(0, tflite_params->stride_height);
            tensor->SetStride(1, tflite_params->stride_width);
            tensor->SetDilation(0, tflite_params->dilation_height_factor);
            tensor->SetDilation(1, tflite_params->dilation_width_factor);
            tensor->SetPaddingType(tflite_params->padding);
            tensor->SetOpType(DEPTHWISECONV2D);
            tensor->SetActType(tflite_params->activation);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            std::vector<uint32_t> inputShape(4, 0);
            for (j = 0; j < 4; ++j)
            {
                inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
            }
            tensor->SetInputShape(inputShape);

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
                // for (int k = 0; k < 30; k++)
                // {
                //     cout << src[k] << endl;
                // }
                // exit(1);
                // MHWI->MIHW
                Transpose(src, weight_bias.first, FILTER_FORMAT_MHWI, FILTER_FORMAT_MIHW, tflite_tensor->dims->data);
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
                // for (int k = 0; k < 30; k++)
                // {
                //     cout << weight_bias.second[k] << endl;
                // }
                // exit(1);
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
            tensor->SetOpType(CONV2D);
            tensor->SetActType(tflite_params->activation);
            // std::cout << tflite_params->activation << std::endl;
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            std::vector<uint32_t> inputShape(4, 0);
            for (j = 0; j < 4; ++j)
            {
                inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
            }
            tensor->SetInputShape(inputShape);
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
                Transpose(src, weight_bias.first, FILTER_FORMAT_OHWI, FILTER_FORMAT_OIHW, tflite_tensor->dims->data);
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

        else if (tflite_op == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR)
        {
            TfLiteResizeNearestNeighborParams *tflite_params = (TfLiteResizeNearestNeighborParams *)node.builtin_data;
            bool align_corners = tflite_params->align_corners;
            bool half_pixel_centers = tflite_params->half_pixel_centers;
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            tensor->SetOpType(RESIZE_NEAREST_NEIGHBOR);
            for (i = 0; i < node.inputs->size - 1; i++) // the error from  parsing tflite resize
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                    // std::cout << inputShape[j] << std::endl;
                }
                tensor->SetInputShape(inputShape);
            }

            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_AVERAGE_POOL_2D)
        {
            TfLitePoolParams *tflite_params = (TfLitePoolParams *)node.builtin_data;
            int stride_width = tflite_params->stride_width;
            int stride_height = tflite_params->stride_height;
            int filter_width = tflite_params->filter_width;
            int filter_height = tflite_params->filter_height;
            tensor->SetStride(0, tflite_params->stride_height);
            tensor->SetStride(1, tflite_params->stride_width);
            tensor->SetKernelShape(0, 0);
            tensor->SetKernelShape(1, 0);
            tensor->SetKernelShape(2, filter_height);
            tensor->SetKernelShape(3, filter_width);
            tensor->SetDilation(0, 1);
            tensor->SetDilation(1, 1);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            tensor->SetOpType(AVERAGE_POOL_2D);
            tensor->SetPaddingType(tflite_params->padding);
            tensor->SetActType(tflite_params->activation);

            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_MAX_POOL_2D)
        {
            TfLitePoolParams *tflite_params = (TfLitePoolParams *)node.builtin_data;
            int stride_width = tflite_params->stride_width;
            int stride_height = tflite_params->stride_height;
            int filter_width = tflite_params->filter_width;
            int filter_height = tflite_params->filter_height;
            tensor->SetStride(0, tflite_params->stride_height);
            tensor->SetStride(1, tflite_params->stride_width);
            tensor->SetKernelShape(0, 0);
            tensor->SetKernelShape(1, 0);
            tensor->SetKernelShape(2, filter_height);
            tensor->SetKernelShape(3, filter_width);
            tensor->SetDilation(0, 1);
            tensor->SetDilation(1, 1);
            tensor->SetActType(tflite_params->activation);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[0]);
            tensor->SetOpType(MAX_POOL_2D);
            tensor->SetPaddingType(tflite_params->padding);
            tensor->SetActType(tflite_params->activation);
            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_CONCATENATION)
        {
            TfLiteConcatenationParams *tflite_params = (TfLiteConcatenationParams *)node.builtin_data;
            tensor->SetActType(tflite_params->activation);
            tensor->SetConcatAxis(tflite_params->axis);
            tensor->SetOpType(CONCATENATION);
            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_ADD)
        {
            TfLiteAddParams *tflite_params = (TfLiteAddParams *)node.builtin_data;
            tensor->SetActType(tflite_params->activation);
            tensor->SetOpType(ADD);
            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                    // std::cout << inputShape[j] << std::endl;
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_SUB)
        {
            TfLiteSubParams *tflite_params = (TfLiteSubParams *)node.builtin_data;
            tensor->SetActType(tflite_params->activation);
            tensor->SetOpType(SUB);
            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_MUL)
        {
            TfLiteMulParams *tflite_params = (TfLiteMulParams *)node.builtin_data;
            tensor->SetActType(tflite_params->activation);
            tensor->SetOpType(MUL);
            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_DIV)
        {
            TfLiteDivParams *tflite_params = (TfLiteDivParams *)node.builtin_data;
            tensor->SetActType(tflite_params->activation);
            tensor->SetOpType(REALDIV);
            for (i = 0; i < node.inputs->size; i++)
            {
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i]);
                std::vector<uint32_t> inputShape(4, 0);
                for (j = 0; j < 4; ++j)
                {
                    inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
                }
                tensor->SetInputShape(inputShape);
            }
            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {

                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
        }
        else if (tflite_op == tflite::BuiltinOperator_TRANSPOSE_CONV)
        {
            TfLiteTransposeConvParams *tflite_params = (TfLiteTransposeConvParams *)node.builtin_data;
            // Default: transpose conv no bias data

            tensor->SetOpType(DECONV2D);
            tensor->SetStride(0, tflite_params->stride_height);
            tensor->SetStride(1, tflite_params->stride_width);
            tensor->SetActType(kActNone);
            tensor->SetPaddingType(tflite_params->padding);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[2]);
            std::vector<uint32_t> inputShape(4, 0);
            for (j = 0; j < 4; ++j)
            {
                inputShape[j] = static_cast<uint32_t>(tflite_tensor->dims->data[j]);
            }
            tensor->SetInputShape(inputShape);

            tflite_tensor = this->tflite_interpreter->tensor(node.outputs->data[0]);
            for (j = 0; j < 4; ++j)
            {
                tensor->SetOutputShape(j, static_cast<uint32_t>(tflite_tensor->dims->data[j]));
            }
            ptr_size = (node.inputs->size - 1) / 2;
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[1]);
            dimSizes = tflite_tensor->dims->data[0] * tflite_tensor->dims->data[1] * tflite_tensor->dims->data[2] * tflite_tensor->dims->data[3];
            weightBytes = dimSizes * sizeof(float);
            tensor->SetWeightBytes(static_cast<uint32_t>(weightBytes));
            float *src = tflite_tensor->data.f;
            weight_bias.first = (float *)malloc(weightBytes);
            Transpose(src, weight_bias.first, FILTER_FORMAT_OHWI, FILTER_FORMAT_OIHW, tflite_tensor->dims->data);
            tensor->SetKernelShape(0, tflite_tensor->dims->data[0]);
            tensor->SetKernelShape(1, tflite_tensor->dims->data[3]);
            tensor->SetKernelShape(2, tflite_tensor->dims->data[1]);
            tensor->SetKernelShape(3, tflite_tensor->dims->data[2]);
            tensor->SetBiasBytes(0);
            weight_bias.second = nullptr;
        }
        tensor->SetMemoryPtr(ptr_size);
        int start_index = mainMemory->size();
        for (j = start_index; j < start_index + ptr_size; j++)
        {
            tensor->SetMemoryPtrIndex(j - start_index, static_cast<uint8_t>(start_index));
        }

        mainMemory->push_back(weight_bias);
    }
    std::vector<std::shared_ptr<Tensor>> Interpreter::mGraphToSNNGraph(std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory,
                                                                       std::map<std::string, std::vector<int>> &modelMaps)
    {
        int node_index, i, j;
        int tensor_index = 0, merged_counts = 0, k = 0;
        std::string op_name;
        std::vector<int> execution_plan = this->tflite_interpreter->execution_plan();
        std::vector<std::shared_ptr<Tensor>> GraphNodes;
        tflite::BuiltinOperator tflite_op;
        numOperators = execution_plan.size() + 1;
        // for predefine constructor
        mainMemory->reserve(numOperators);
        GraphNodes.reserve(numOperators);
        printf("INFO: Start converting Tf-Lite graph to SNN graph ... \n");
        std::map<int, int> tflite_infos;
        this->GetInputTensor(GraphNodes, modelMaps);
        std::shared_ptr<Tensor> last_tensor;
        int outputIndexSize = tflite_interpreter->outputs().size();
        std::vector<int> outputIndex(outputIndexSize, 0);
        for (auto &node_index : execution_plan)
        {
            // printf("---------------------- node index is %d ---------------------\n", tensor_index);
            std::vector<int> index;
            const std::pair<TfLiteNode, TfLiteRegistration> *node_and_registration = (this->tflite_interpreter)->node_and_registration(node_index);
            const TfLiteNode &node = node_and_registration->first;
            const TfLiteRegistration &registration = node_and_registration->second;
            op_name = tflite::GetOpNameByRegistration(registration);
            tflite_op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
            for (j = 0; j < outputIndexSize; j++)
            {
                // cout << tflite_interpreter->outputs()[j] << endl;
                if (tflite_interpreter->outputs()[j] == node.outputs->data[0])
                {
                    outputIndex[k] = tensor_index + 1;
                    k++;
                }
            }
            // for transpose convolution
            if ((tflite_op == tflite::BuiltinOperator_SHAPE) ||
                (tflite_op == tflite::BuiltinOperator_STRIDED_SLICE) ||
                (tflite_op == tflite::BuiltinOperator_PACK))
                continue;
            if (tflite_op == tflite::BuiltinOperator_LOGISTIC)
            {
                last_tensor = GraphNodes.at(tensor_index);
                last_tensor->SetActType(TfLiteFusedActivation::kTfLiteActSigmoid);
                merged_counts++;
                continue;
            }

            // new tensor for snn graph
            std::shared_ptr<Tensor> tensor(new Tensor()); // default tensor
            tensor->SetOpName(op_name);
            tflite_infos[node.outputs->data[0]] = tensor_index;
            if (tflite_op == tflite::BuiltinOperator_CONCATENATION ||
                tflite_op == tflite::BuiltinOperator_ADD ||
                tflite_op == tflite::BuiltinOperator_MUL ||
                tflite_op == tflite::BuiltinOperator_DIV ||
                tflite_op == tflite::BuiltinOperator_MUL)
            {
                index = {tflite_infos[node.inputs->data[0]] + 1, tflite_infos[node.inputs->data[1]] + 1};
            }
            else
            {
                if (node_index == 0)
                {
                    index = {tflite_infos[node.inputs->data[0]]};
                }
                else
                {
                    index = {tflite_infos[node.inputs->data[0] + 1]};
                }
            }
            tensor->inputIndex = index;
            this->IdentifyOperation(tensor, mainMemory, node, tflite_op);
            GraphNodes.emplace_back(tensor);
            tensor_index++;
        }
        numOperators -= merged_counts;
        SNN_ASSERT(tensor_index + 1 == numOperators);
        GraphNodes.resize(numOperators);
        modelMaps["outputIndex"] = outputIndex;
        tflite_interpreter.reset();
        TFLITE_MINIMAL_CHECK((tflite_interpreter) == nullptr);
        return GraphNodes;
    }
    void Interpreter::GetInputTensor(std::vector<std::shared_ptr<Tensor>> &GraphNodes, std::map<std::string, std::vector<int>> &modelMaps)
    {
        TfLiteTensor *tflite_tensor = this->tflite_interpreter->input_tensor(0);
        std::shared_ptr<Tensor> input_tensor(new Tensor()); // default tensor
        std::vector<uint32_t> inputShape(4, 0);
        int i;
        for (i = 0; i < 4; ++i)
        {
            inputShape[i] = static_cast<uint32_t>(tflite_tensor->dims->data[i]);
            input_tensor->SetOutputShape(i, static_cast<uint32_t>(tflite_tensor->dims->data[i]));
        }
        int inputSize = tflite_interpreter->inputs().size();
        std::vector<int> inputIndex(inputSize, 0);
        for (i = 0; i < inputSize; i++)
        {
            inputIndex[i] = tflite_interpreter->inputs()[i];
        }
        input_tensor->SetInputShape(inputShape);
        input_tensor->SetOpType(INPUTDATA);
        GraphNodes.emplace_back(input_tensor);
        modelMaps["inputIndex"] = inputIndex;
    }
    void Interpreter::Transpose(float *src, float *dst, FilterFormat inFormat, FilterFormat outFormat, int *shapDims)
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
                            // printf("%f \n", src[fromIndex]);
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