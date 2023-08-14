#include "Interpreter.h"
namespace SNN
{
    Interpreter::Interpreter(string model_path)
    {
        // device initialize and model
        threads = -1;
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        TFLITE_MINIMAL_CHECK(model != nullptr);
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        unique_ptr<tflite::Interpreter> tflite_interpreter(new tflite::Interpreter());
        builder.PreserveAllTensorsExperimental();
        builder(&this->tflite_interpreter, threads);
        TFLITE_MINIMAL_CHECK((this->tflite_interpreter) != nullptr);
        backendfactory = new BackendFactory(false);
    }
    Interpreter::~Interpreter() {}

    void Interpreter::IdentifyOperation(Tensor *snn_tensor, const TfLiteNode &node, tflite::BuiltinOperator tflite_op, int *kernel_dims)
    {
        TfLiteTensor *tflite_tensor;
        int i;
        int value;
        switch (tflite_op)
        {
        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        {
            if (node.inputs->size != 3)
            {
                printf("Error only for dimension 3\n");
                exit(1);
            }
            TfLiteDepthwiseConvParams *tflite_params = (TfLiteDepthwiseConvParams *)node.builtin_data;
            DepthwiseConvParams *snn_params = (DepthwiseConvParams *)malloc(sizeof(DepthwiseConvParams));
            snn_params->stride_height = tflite_params->stride_height;
            snn_params->stride_width = tflite_params->stride_width;
            snn_params->dilation_height_factor = tflite_params->dilation_height_factor;
            snn_params->dilation_width_factor = tflite_params->dilation_width_factor;
            snn_params->depth_multiplier = tflite_params->depth_multiplier;
            snn_params->padding = (Padding)tflite_params->padding;
            snn_params->activation = (FusedActivation)tflite_params->activation;
            snn_tensor->op_type = DepthwiseConv;
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[1]);
            for (i = 0; i < 4; ++i)
                snn_params->kernel_dims[i] = tflite_tensor->dims->data[i];

            snn_params->weight_bytes = tflite_tensor->bytes;
            snn_params->weights = (float *)malloc(snn_params->weight_bytes);
            memcpy(snn_params->weights, tflite_tensor->data.f, snn_params->weight_bytes);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[2]);
            snn_params->bias_dims[0] = tflite_tensor->dims->data[0];
            snn_params->bias_bytes = tflite_tensor->bytes;
            snn_params->bias = (float *)malloc(snn_params->bias_bytes);
            memcpy(snn_params->bias, tflite_tensor->data.f, snn_params->bias_bytes);
            snn_tensor->op_data = (void *)snn_params;
        }
        case tflite::BuiltinOperator_CONV_2D:
        {
            if (node.inputs->size != 3)
            {
                printf("Error only for dimension 3\n");
                exit(1);
            }
            TfLiteConvParams *tflite_params = (TfLiteConvParams *)node.builtin_data;
            ConvParams *snn_params = (ConvParams *)malloc(sizeof(ConvParams));
            snn_params->stride_height = tflite_params->stride_height;
            snn_params->stride_width = tflite_params->stride_width;
            snn_params->dilation_height_factor = tflite_params->dilation_height_factor;
            snn_params->dilation_width_factor = tflite_params->dilation_width_factor;
            snn_params->padding = (Padding)tflite_params->padding;
            snn_params->activation = (FusedActivation)tflite_params->activation;
            snn_tensor->op_type = DepthwiseConv;
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[1]);
            for (i = 0; i < 4; ++i)
                snn_params->kernel_dims[i] = tflite_tensor->dims->data[i];
            snn_params->weight_bytes = tflite_tensor->bytes;
            snn_params->weights = (float *)malloc(snn_params->weight_bytes);
            memcpy(snn_params->weights, tflite_tensor->data.f, snn_params->weight_bytes);
            tflite_tensor = this->tflite_interpreter->tensor(node.inputs->data[2]);
            snn_params->bias_bytes = tflite_tensor->bytes;
            snn_params->bias = (float *)malloc(snn_params->bias_bytes);
            memcpy(snn_params->bias, tflite_tensor->data.f, snn_params->bias_bytes);
            snn_tensor->op_data = (void *)snn_params;
        }
        }
    }
    vector<SNNNode> Interpreter::TfliteGraphToSNNGraph()
    {
        int node_index, i, j, k;
        vector<int> execution_plan = this->tflite_interpreter->execution_plan();
        vector<SNNNode> GraphNodes;
        tflite::BuiltinOperator tflite_op;
        TfLiteTensor *tflite_tensor;
        printf("INFO: Start converting Tf-Lite nodes to SNN nodes ... \n");
        for (i = 0; i < execution_plan.size(); ++i)
        {
            node_index = execution_plan[i];
            SNNNode snnnode;
            const pair<TfLiteNode, TfLiteRegistration> *node_and_registration = (this->tflite_interpreter)->node_and_registration(node_index);
            const TfLiteNode &node = node_and_registration->first;
            const TfLiteRegistration &registration = node_and_registration->second;
            tflite_op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
            string op_name = tflite::GetOpNameByRegistration(registration);
            snnnode.inputs = (IntArray *)malloc(sizeof(IntArray));
            snnnode.outputs = (IntArray *)malloc(sizeof(IntArray));
            snnnode.tensor = (Tensor *)malloc(sizeof(Tensor));
            snnnode.inputs->size = node.inputs->size;
            snnnode.inputs->data[node.inputs->size];
            snnnode.outputs->size = node.outputs->size;
            snnnode.outputs->data[node.outputs->size];
            IdentifyOperation(snnnode.tensor, node, tflite_op, tflite_tensor->dims->data);
            snnnode.tensor->device_type = OpenCL;
            this->backendfactory->BuildOperation(snnnode.tensor);
            exit(1);

            for (j = 0; j < node.outputs->size; ++j)
                snnnode.outputs->data[j] = node.outputs->data[j];

            // else if (tflite_op == tflite::BuiltinOperator_CONV_2D)
            // {
            //     TfLiteConvParams *params = (TfLiteConvParams *)node.builtin_data;
            //     snnnode.tensor->op_data = (TfLiteConvParams *)malloc(sizeof(TfLiteConvParams));
            //     snnnode.tensor->op_type = Conv;
            // }
        }
    }
    void Interpreter::GetInferGraph()
    {
    }

    void Interpreter::EnableOpenCL()
    {
        backendfactory->SetOpenCLBackend();
    }

}