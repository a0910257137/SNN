#ifndef MODEL_H
#define MODEL_H
#include "include/SNN/common.h"
#include "NonCopyable.h"
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include "Interpreter.h"
#include "misc/utils.h"
#include "PostProcessor.h"
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
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> Inference(float *input_data, float *resizedRatios);

    protected:
        std::shared_ptr<Backend> backend;

    public:
        std::vector<std::shared_ptr<Execution>> netOpContainer;

    private:
        bool msupportModel = false;
        std::string inputModelFormat;
        std::unique_ptr<PostProcessor> mpostProcessor;
        std::unique_ptr<Interpreter> interpreter;
        std::vector<std::shared_ptr<Tensor>> snnGraph;
        std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory;
        std::map<std::string, std::map<std::string, std::vector<int>>> mModelMaps;
        std::string modelName;
    };
}
#endif /* MODEL_H */
