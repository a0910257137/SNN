#ifndef INTERPRETER_H
#define INTERPRETER_H
#include <string>
#include <unistd.h>
#include <iostream>
#include "misc/utils.h"
#include "include/SNN/Tensor.h"
#include "backend/Backend.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }
namespace SNN
{
    using namespace std;
    class Interpreter
    {
    public:
        Interpreter(int batchSize, string model_path);
        ~Interpreter();
        Interpreter(const Interpreter &) = delete;
        Interpreter &operator=(const Interpreter &) = delete;
        vector<shared_ptr<Tensor>> mGraphToSNNGraph(std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory,
                                                    std::map<std::string, std::vector<int>> &modelMaps);

    private:
        void IdentifyOperation(shared_ptr<Tensor> tensor,
                               std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory,
                               const TfLiteNode &tflite_params,
                               tflite::BuiltinOperator tflite_op);
        void GetInputTensor(std::vector<std::shared_ptr<Tensor>> &GraphNodes, std::map<std::string, std::vector<int>> &modelMaps);
        void Transpose(float *src, float *dst, FilterFormat inFormat, FilterFormat outFormat, int *shapDims);

    public:
        int numOperators;

    private:
        int threads;
        int batchSize = 2;
        unique_ptr<tflite::FlatBufferModel> model;
        unique_ptr<tflite::Interpreter> tflite_interpreter;
        Backend *backend = nullptr;
    };

} // namespace  SNN

#endif