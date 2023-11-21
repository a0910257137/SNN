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
        modelName = remove_extension(base_name(model_cfg.mtfd_path));
        std::map<std::string, std::vector<int>> baseInfos;
        std::vector<int> inputIndex, outputIndex;
        baseInfos["inputIndex"] = inputIndex;
        baseInfos["outputIndex"] = outputIndex;
        mModelMaps[modelName] = baseInfos;
        mpostProcessor = std::make_unique<PostProcessor>(model_cfg);
        interpreter = std::make_unique<Interpreter>(model_cfg.mtfd_path);
        mainMemory = std::make_shared<std::vector<std::pair<float *, float *>>>();
    }
    Model::~Model()
    {
        mModelMaps.clear();
        snnGraph.clear();
    }
    bool Model::GetSNNGraph()
    {
        snnGraph = interpreter->mGraphToSNNGraph(mainMemory, mModelMaps[modelName]);

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
        std::string name;
        std::shared_ptr<Tensor> tensor;
        for (int i = 1; i < interpreter->numOperators; i++)
        {
            tensor = snnGraph[i];
            tensor->SetMainMemory(mainMemory);
            // name = tensor->GetOpName();
            this->backend->BuildOperation(tensor, netOpContainer);
        }
        return true;
    }

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> Model::Inference(float *input_data, float *resizedRatios)
    {
        int i, j;
        this->backend->ConvertInputBuffer(snnGraph.at(0), input_data);
        bool status;
        std::shared_ptr<Execution> op;
        std::vector<int> input_idx, output_idx;
        std::vector<std::shared_ptr<Tensor>> inputs, outputs;
        inputs.reserve(3);
        outputs.reserve(3);
        std::shared_ptr<Tensor> tensor;
        // std::string name;
        for (i = 0; i < interpreter->numOperators - 1; i++)
        {
            // std::cout << " ------------------------ Node op index: ------------------------  " << i << std::endl;
            tensor = snnGraph.at(i + 1);
            // name = tensor->GetOpName();
            const std::vector<int> &inputIndex = tensor->inputIndex;
            op = netOpContainer[i];
            for (j = 0; j < inputIndex.size(); j++)
            {
                inputs.emplace_back(snnGraph.at(inputIndex[j]));
            }
            outputs.emplace_back(snnGraph.at(i + 1));
            status = op->onExecute(inputs, outputs);
            inputs.clear();
            outputs.clear();
        }
        std::vector<int> &inputIndex = mModelMaps[modelName]["inputIndex"];
        std::vector<int> &outputIndex = mModelMaps[modelName]["outputIndex"];
        float *cls_x, *bbox_x, *x;
        /** Predicting outputs
         * "s0_x", "s0_bbox_x", "s0_cls_x"
         * "s1_x", "s1_bbox_x", "s1_cls_x"
         * "s2_x", "s2_bbox_x", "s2_cls_x"
         * */
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> results;
        for (i = 0; i < outputIndex.size() / 3; i++)
        {
            // std::cout << " ------------------------ Node op index: ------------------------  " << std::endl;
            tensor = snnGraph.at(outputIndex[3 * i]);
            const std::vector<int> &xShape = tensor->OutputShape();
            x = op->onConvert(tensor);
            tensor = snnGraph.at(outputIndex[3 * i + 1]);
            const std::vector<int> &bbox_xShape = tensor->OutputShape();
            bbox_x = op->onConvert(tensor);
            tensor = snnGraph.at(outputIndex[3 * i + 2] - 1);
            const std::vector<int> &cls_xShape = tensor->OutputShape();
            cls_x = op->onConvert(tensor);
            mpostProcessor->MTFDProcessor(results,
                                          i,
                                          resizedRatios,
                                          cls_x,
                                          bbox_x,
                                          x,
                                          xShape,
                                          bbox_xShape,
                                          cls_xShape);
            free(x);
            free(cls_x);
            free(bbox_x);
        }
        this->backend->ReleaseBuffer(snnGraph.at(0));
        return results;
    }
} // namespace SNN
