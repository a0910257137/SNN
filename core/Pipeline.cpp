#include "Pipeline.h"
namespace SNN
{
    Pipeline::Pipeline(std::string model_path, BackendConfig &cfg)
    {

        char *ptr;
        char arr[model_path.length() + 1];
        strcpy(arr, model_path.c_str());
        ptr = std::strtok(arr, ".");
        std::string s_ptr;
        for (int i = 0; i < 2; i++)
        {
            s_ptr = static_cast<std::string>(ptr);
            if (s_ptr == "tflite")
            {
                msupportModel = true;
                inputModelFormat = s_ptr;
                printf("INFO: The format of input model is tflite\n");
            }

            ptr = strtok(NULL, " . ");
        }

        if (!msupportModel)
        {
            std::cout << "ERROR: The" << inputModelFormat << "model format are not supported in SNN " << SNN_VERSION << "!!" << std::endl;
            exit(1);
        }
        nodefactory = std::make_shared<NodeFactory>(cfg);
        interpreter = std::make_unique<Interpreter>(model_path);
        mainMemory = std::make_shared<std::vector<std::pair<float *, float *>>>();
    }
    Pipeline::~Pipeline()
    {
    }
    bool Pipeline::GetSNNGraph()
    {
        snnGraph = interpreter->mGraphToSNNGraph(mainMemory);
        if (snnGraph.size() == 0)
        {
            printf("ERROR: Gernerated empty SNN graph by %s\n", inputModelFormat.c_str());
            SNN_CHECK_SUCCESS(snnGraph.size() != 0, true);
        }
        else
            printf("INFO: Finsh converting to SNN nodes ... \n");
    }
    bool Pipeline::BuildSNNGraph()
    {

        bool status = true;
        firstMallocs = std::shared_ptr<bool[]>(new bool[10]);
        int i = 0;
        for (std::shared_ptr<Tensor> tensor : snnGraph)
        {
            tensor->SetMainMemory(mainMemory);
            status |= nodefactory->BuildOperation(tensor);
            if (status == false)
                printf("ERROR: build error !! \n");
            i++;
        }

        return status;
    }

} // namespace  SNN
