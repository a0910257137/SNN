#include "Model.h"

namespace SNN
{
    Model::Model(ModelConfig &model_cfg)
    {
        char *ptr;
        char arr[model_cfg.mtfd_path.length() + 1];
        strcpy(arr, model_cfg.mtfd_path.c_str());
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
        // add hint error file name
        if (!msupportModel)
        {
            std::cout << "ERROR: The" << inputModelFormat << "model format are not supported in SNN " << SNN_VERSION << "!!" << std::endl;
            exit(1);
        }
        interpreter = std::make_unique<Interpreter>(model_cfg.mtfd_path);
        mainMemory = std::make_shared<std::vector<std::pair<float *, float *>>>();
        // mainMemory->reserve
    }
    Model::~Model()
    {
        // printf("Release\n");
        snn_infos.clear();
    }
    bool Model::GetSNNGraph()
    {
        snnGraph = interpreter->mGraphToSNNGraph(mainMemory, snn_infos);
        if (snnGraph.size() == 0)
        {
            printf("ERROR: Gernerated empty SNN graph by %s\n", inputModelFormat.c_str());
            SNN_CHECK_SUCCESS(snnGraph.size() != 0, true);
        }
        else
            printf("INFO: Finsh converting to SNN nodes ... \n");
    }
    bool Model::BuildSNNGraph()
    {
        netOpContainer.reserve(interpreter->numOperators - 1);
        for (int i = 1; i < interpreter->numOperators; i++)
        {
            std::shared_ptr<Tensor> tensor = snnGraph[i];
            tensor->SetMainMemory(mainMemory);
            this->backend->BuildOperation(tensor, netOpContainer);
        }
        return true;
    }

    bool Model::Inference(float *input_data)
    {
        int i, j;
        this->backend->ConvertInputBuffer(snnGraph.at(0), input_data);
        bool status;
        std::shared_ptr<Execution> op;
        std::vector<int> input_idx, output_idx;
        for (i = 0; i < interpreter->numOperators - 1; i++)
        {
            //     // std::cout << " ------------------------ Node op index: ------------------------  " << i << std::endl;
            std::vector<std::shared_ptr<Tensor>> inputs, outputs;
            inputs.reserve(2);
            outputs.reserve(2);
            op = netOpContainer[i];
            input_idx = snn_infos[i];
            for (j = 0; j < input_idx.size(); j++)
            {
                inputs.push_back(snnGraph.at(input_idx[j]));
            }
            outputs.push_back(snnGraph.at(i + 1));
            status = op->onExecute(inputs, outputs);
        }
        // op->onConvert(snnGraph.at(121));
        this->backend->ReleaseBuffer(snnGraph.at(0));
        
        // exit(1);
    }
} // namespace SNN
