#include "Interpreter.h"
namespace SNN
{
    Interpreter::Interpreter(string model_path, bool enable_gpu)
    {
        // device initialize and model
        threads = -1;
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        TFLITE_MINIMAL_CHECK(model != nullptr);
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        unique_ptr<tflite::Interpreter> tflite_interpreter(new tflite::Interpreter());
        builder.PreserveAllTensorsExperimental();
        builder(&tflite_interpreter, threads);
        TFLITE_MINIMAL_CHECK(tflite_interpreter != nullptr);
        // TFLITE_MINIMAL_CHECK(tflite_interpreter->AllocateTensors() == kTfLiteOk);
        if (enable_gpu)
        {
            bool is_half = true;
            const OpenCLBackend *K_backend = new OpenCLBackend(is_half);
            exit(1);
        }
    }
    void Interpreter::constructGraph()
    {
        vector<int> execution_plan = tflite_interpreter->execution_plan();
        int node_index, i, j, k;
        for (i = 0; i < execution_plan.size(); ++i)
        {
            node_index = execution_plan[i];
            const pair<TfLiteNode, TfLiteRegistration> *node_and_registration = tflite_interpreter->node_and_registration(node_index);
            const TfLiteNode &node = node_and_registration->first;
            const TfLiteRegistration &registration = node_and_registration->second;
            for (int i = 0; i < node.inputs->size; ++i)
            {
                int tensor_index = node.inputs->data[i];
                TfLiteTensor *tensor = tflite_interpreter->tensor(tensor_index);
                TfLiteType dtype = tensor->type;
                TfLiteIntArray *dims = tensor->dims;
                string s = tensor->name;
                if (s == "serving_default_image_inputs:0")
                {
                    continue;
                }
                int bytes = tensor->bytes;
                int length = bytes / sizeof(float);
                cout << "The tensor are " << bytes << " btyes; " << endl;
                cout
                    << "The " << s << " tensor dimensions are ";
                for (int i = 0; i < dims->size; ++i)
                {
                    cout << dims->data[i] << " ";
                }
                cout << endl;
                if (dtype == kTfLiteFloat32)
                {
                    float *data = tensor->data.f;
                    for (int i = 0; i < length; ++i)
                    {
                        cout << *data << endl;
                        data++;
                    }
                }
            }
        }
    }
    Interpreter::~Interpreter() {}
}