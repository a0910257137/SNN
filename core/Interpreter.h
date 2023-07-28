#ifndef INTERPRETER_h
#define INTERPRETER_h
#include <string>
#include <unistd.h>
#include <iostream>
#include "misc/utils.h"
#include "include/SNN/api_types.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/core/subgraph.h"

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
        Interpreter(string model_path, bool enable_gpu);
        ~Interpreter();
        Interpreter(const Interpreter &) = delete;
        Interpreter &operator=(const Interpreter &) = delete;
        void constructGraph();

    private:
        unique_ptr<tflite::FlatBufferModel> model;
        unique_ptr<tflite::Interpreter> tflite_interpreter;
        int threads;
    };

} // namespace  SNN

#endif