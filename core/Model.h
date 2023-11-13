#ifndef MODEL_H
#define MODEL_H
#include "include/SNN/common.h"
#include "NonCopyable.h"
#include <iostream>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>
#include "Interpreter.h"

namespace SNN
{

    class Model : public NonCopyable
    {
    public:
        Model(ModelConfig &model_cfg);
        ~Model();

    public:
        bool GetSNNGraph();
        bool BuildSNNGraph();
        bool Inference(float *input_data);
        

    protected:
        std::shared_ptr<Backend> backend;

    public:
        std::vector<std::shared_ptr<Execution>> netOpContainer;
        std::map<int, std::vector<int>> snn_infos;

    private:
        bool mAllocInput;
        bool mOutputStatic;
        bool msupportModel = false;
        std::string inputModelFormat;
        std::unique_ptr<Interpreter> interpreter;
        std::vector<std::shared_ptr<Tensor>> snnGraph;
        std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory;
    };
}
#endif /* MODEL_H */
