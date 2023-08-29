#include <math.h>
#include <string>
#include <time.h>
#include <algorithm>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <vector>
#include "macro.h"
#include "core/Interpreter.h"
#include "core/Pipeline.h"
using namespace std;
// auto tf_tensor_to_vector(tensorflow::Tensor tensor, int32_t tensorSize)
// {
//     int32_t *tensor_ptr = tensor.flat<int32_t>().data();
//     std::vector<int32_t> v(tensor_ptr, tensor_ptr + tensorSize);
//     return v;
// }

int main(int argc, char **argv)
{
    string path = "/aidata/anders/data_collection/okay/WF/archives/test/FP32.tflite";
    SNN::Pipeline pipeline(path, OpenCL);
    bool status;
    status = pipeline.GetSNNGraph();
    status = pipeline.BuildSNNGraph();
    return 1;
    // interpreter.GetInferGraph();
    // unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    // if (!model)
    //     cerr << "Failed to mmap tflite model" << endl;
    // TFLITE_MINIMAL_CHECK(model != nullptr);
    // tflite::ops::builtin::BuiltinOpResolver resolver;
    // tflite::InterpreterBuilder builder(*model, resolver);
    // unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter());
    // builder.PreserveAllTensorsExperimental();
    // builder(&interpreter, 5);
    // TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    // TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // string image_path = "/aidata/anders/data_collection/okay/300W/imgs/lfpw_image_0373.png.jpg";
    // cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    // cv::Mat resized_image;
    // cv::resize(image, resized_image, cv::Size(640, 640), 0, 0, CV_INTER_LINEAR);
    // const int kInputBytes = sizeof(uchar) * 640 * 640 * 3;
    // float *input_img = interpreter->typed_tensor<float>(0);
    // std::memcpy((uchar *)input_img, resized_image.data, kInputBytes);
    // vector<int> execution_plan = interpreter->execution_plan();
    // int node_index;
    // int i, j;
    // for (i = 0; i < execution_plan.size(); ++i)
    // {
    //     if (i > 0)
    //     {
    //         continue;
    //     }
    //     node_index = execution_plan[i];
    //     const pair<TfLiteNode, TfLiteRegistration> *node_and_registration = interpreter->node_and_registration(node_index);
    //     const TfLiteNode &node = node_and_registration->first;
    //     const TfLiteRegistration &registration = node_and_registration->second;
    //     print(80);
    //     for (int i = 0; i < node.inputs->size; ++i)
    //     {
    //         int tensor_index = node.inputs->data[i];
    //         TfLiteTensor *tensor = interpreter->tensor(tensor_index);
    //         TfLiteType dtype = tensor->type;
    //         TfLiteIntArray *dims = tensor->dims;
    //         string s = tensor->name;
    //         if (s == "serving_default_image_inputs:0")
    //         {
    //             continue;
    //         }

    //         int bytes = tensor->bytes;
    //         int length = bytes / sizeof(float);
    //         cout << "The tensor are " << bytes << " btyes; " << endl;
    //         cout
    //             << "The " << s << " tensor dimensions are ";
    //         for (int i = 0; i < dims->size; ++i)
    //         {
    //             cout << dims->data[i] << " ";
    //         }
    //         cout << endl;
    //         if (dtype == kTfLiteFloat32)
    //         {
    //             float *data = tensor->data.f;
    //             for (int i = 0; i < length; ++i)
    //             {
    //                 cout << *data << endl;
    //                 data++;
    //             }
    //         }
    //         // exit(1);
    //     }

    // for (int i = 0; i < 80; ++i)
    // {
    //     printf("-");
    // }
    // printf("\n");
    // for (int i = 0; i < node.outputs->size; ++i)
    // {
    //     int tensor_index = node.outputs->data[i];
    //     TfLiteTensor *tensor = interpreter->tensor(tensor_index);
    //     TfLiteType dtype = tensor->type;
    //     TfLiteIntArray *dims = tensor->dims;
    //     string s = tensor->name;
    //     cout << "The " << s << " tensor dimensions are ";
    //     for (int i = 0; i < dims->size; ++i)
    //     {
    //         cout << dims->data[i] << " ";
    //     }
    //     cout << "" << endl;
    // }
    // }
    // interpreter->Invoke();
    // return 1;
}