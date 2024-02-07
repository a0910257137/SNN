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
        static cv::Mat frame, demo_frame;
        static pthread_mutex_t lock_video_in, lock_model, lock_show;
        static pthread_cond_t cond_video_in, cond_model, cond_show;
        static bool status_video, status_model, status_show;
        static bool exitSignal;
        static std::vector<std::shared_ptr<Tensor>> MTFDGraph;
        static std::vector<std::shared_ptr<Execution>> MTFDOpContainer;

        static std::vector<std::shared_ptr<Execution>> optMTFDOpContainer;
        static std::vector<std::vector<std::shared_ptr<Tensor>>> optMTFDGraph;
        static std::map<int, std::vector<int>> optMTFDGraphLinks;

        static bool enableOptimization;
        static float *resizedRatios;
        static float *batchBuffer;
        static int bCounts;
        static int nodeLength;
        static ModelConfig thread_model_cfg;
        static std::vector<int> outputIndex;

    public:
        Model(ModelConfig &model_cfg);
        ~Model();

    public:
        bool GetSNNGraph();
        bool BuildSNNGraph(bool is_optimization = false);
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> Inference(cv::VideoCapture &cap, float *resizedRatios);
        void GraphOptimization();
        static void *InputPreprocess(void *args);
        static void *OperatorInference(void *args);
        static void *Display(void *args);

    protected:
        std::shared_ptr<Backend> backend;
        

    public:
        std::vector<std::shared_ptr<Execution>> netOpContainer;
        std::vector<std::shared_ptr<Tensor>> snnGraph;
        std::vector<std::shared_ptr<Execution>> optOpContainer;
        std::vector<std::vector<std::shared_ptr<Tensor>>> optGraph;
        std::map<int, std::vector<int>> connection_infos;

    private:
        bool msupportModel = false;
        std::string inputModelFormat;
        std::unique_ptr<PostProcessor> mpostProcessor;
        std::unique_ptr<Interpreter> interpreter;
        std::shared_ptr<std::vector<std::pair<float *, float *>>> mainMemory;
        std::map<std::string, std::map<std::string, std::vector<int>>> mModelMaps;
        std::string modelName;
    };
}
#endif /* MODEL_H */
