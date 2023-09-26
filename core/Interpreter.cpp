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
        if (tflite_op == tflite::BuiltinOperator_DEPTHWISE_CONV_2D)
        {
            TfLiteDepthwiseConvParams *tflite_params = (TfLiteDepthwiseConvParams *)node.builtin_data;
            tensor->SetStride(0, static_cast<uint8_t>(tflite_params->stride_height));
            tensor->SetStride(1, static_cast<uint8_t>(tflite_params->stride_width));
            tensor->SetDilation(0, static_cast<uint8_t>(tflite_params->dilation_height_factor));
            tensor->SetDilation(1, static_cast<uint8_t>(tflite_params->dilation_width_factor));
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
                // alloc weights and bias memory
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 1]);
                tensor->SetWeightBytes(static_cast<uint32_t>(tflite_tensor->bytes));
                weight_bias.first = (float *)malloc(tflite_tensor->bytes);
                for (j = 0; j < 4; ++j)
                {
                    tensor->SetKernelShape(j, static_cast<uint8_t>(tflite_tensor->dims->data[j]));
                }
                memcpy(weight_bias.first, tflite_tensor->data.f, tflite_tensor->bytes);
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 2]);
                tensor->SetBiasBytes(static_cast<uint32_t>(tflite_tensor->bytes));
                weight_bias.second = (float *)malloc(tflite_tensor->bytes);
                memcpy(weight_bias.second, tflite_tensor->data.f, tflite_tensor->bytes);
                tensor->SetBiasShape(0, static_cast<uint8_t>(tflite_tensor->dims->data[0]));
            }

            tensor->SetMemoryPtr(ptr_size);
            int start_index = mainMemory->size();
            for (j = start_index; j < start_index + ptr_size; j++)
            {
                tensor->SetMemoryPtrIndex(j - start_index, (uint8_t)start_index);
            }
            mainMemory->push_back(weight_bias);
        }
        else if (tflite_op == tflite::BuiltinOperator_CONV_2D)
        {
            TfLiteConvParams *tflite_params = (TfLiteConvParams *)node.builtin_data;
            tensor->SetStride(0, static_cast<uint8_t>(tflite_params->stride_height));
            tensor->SetStride(1, static_cast<uint8_t>(tflite_params->stride_width));
            tensor->SetDilation(0, static_cast<uint8_t>(tflite_params->dilation_height_factor));
            tensor->SetDilation(1, static_cast<uint8_t>(tflite_params->dilation_width_factor));
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
                // alloc weights and bias memory
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 1]);
                tensor->SetWeightBytes(static_cast<uint32_t>(tflite_tensor->bytes));
                weight_bias.first = (float *)malloc(tflite_tensor->bytes);
                for (j = 0; j < 4; ++j)
                {
                    tensor->SetKernelShape(j, static_cast<uint8_t>(tflite_tensor->dims->data[j]));
                }
                memcpy(weight_bias.first, tflite_tensor->data.f, tflite_tensor->bytes);
                tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[i * 2 + 2]);
                tensor->SetBiasBytes(static_cast<uint32_t>(tflite_tensor->bytes));
                weight_bias.second = (float *)malloc(tflite_tensor->bytes);
                memcpy(weight_bias.second, tflite_tensor->data.f, tflite_tensor->bytes);
                tensor->SetBiasShape(0, static_cast<uint8_t>(tflite_tensor->dims->data[0]));
            }
            tensor->SetMemoryPtr(ptr_size);
            int start_index = mainMemory->size();
            for (j = start_index; j < start_index + ptr_size; j++)
            {
                tensor->SetMemoryPtrIndex(j - start_index, static_cast<uint8_t>(start_index));
            }
            mainMemory->push_back(weight_bias);
        }
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
            for (j = 0; j < node.inputs->size; j++)
                tensor->inputIndex.push_back(node.inputs->data[i]);
            for (j = 0; j < node.outputs->size; j++)
                tensor->outputIndex.push_back(node.outputs->data[i]);
            IdentifyOperation(tensor, mainMemory, node, tflite_op);
            GraphNodes.push_back(tensor);
        }
        return GraphNodes;
    }

}