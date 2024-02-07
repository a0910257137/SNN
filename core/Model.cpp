#include "Model.h"

namespace SNN
{
    cv::Mat Model::frame, Model::demo_frame;
    pthread_mutex_t Model::lock_video_in, Model::lock_model, Model::lock_show;
    pthread_cond_t Model::cond_video_in, Model::cond_model, Model::cond_show;
    bool Model::status_video = true, Model::status_model = false, Model::status_show = false, Model::exitSignal = false, Model::enableOptimization = false;
    std::vector<std::shared_ptr<Tensor>> Model::MTFDGraph;
    std::vector<std::shared_ptr<Execution>> Model::MTFDOpContainer;
    std::vector<std::shared_ptr<Execution>> Model::optMTFDOpContainer;
    std::vector<std::vector<std::shared_ptr<Tensor>>> Model::optMTFDGraph;
    std::map<int, std::vector<int>> Model::optMTFDGraphLinks;
    int Model::nodeLength, Model::bCounts;
    float *Model::batchBuffer, *Model::resizedRatios;
    std::vector<int> Model::outputIndex;
    ModelConfig Model::thread_model_cfg;
    Model::Model(ModelConfig &model_cfg)
    {
        pthread_mutex_init(&lock_video_in, 0);
        pthread_cond_init(&cond_video_in, 0);
        pthread_mutex_init(&lock_model, 0);
        pthread_cond_init(&cond_model, 0);
        pthread_mutex_init(&lock_show, 0);
        pthread_cond_init(&cond_show, 0);
        thread_model_cfg = model_cfg;
        bCounts = model_cfg.batch_size;
        resizedRatios = (float *)malloc(2 * sizeof(float));
        resizedRatios[0] = 1280 / 320;
        resizedRatios[1] = 720 / 320;
        batchBuffer = (float *)malloc(model_cfg.batch_size * 320 * 320 * 3);
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
        interpreter = std::make_unique<Interpreter>(model_cfg.batch_size, model_cfg.mtfd_path);
        mainMemory = std::make_shared<std::vector<std::pair<float *, float *>>>();
    }
    Model::~Model()
    {
        mModelMaps.clear();
        snnGraph.clear();
        pthread_mutex_destroy(&lock_video_in);
        pthread_mutex_destroy(&lock_model);
        pthread_mutex_destroy(&lock_show);
        pthread_cond_destroy(&cond_video_in);
        pthread_cond_destroy(&cond_model);
        pthread_cond_destroy(&cond_show);
        free(resizedRatios);
        free(batchBuffer);
    }
    bool Model::GetSNNGraph()
    {
        snnGraph = interpreter->mGraphToSNNGraph(mainMemory, mModelMaps[modelName]);
        if (snnGraph.size() == 0)
        {
            printf("ERROR: Gernerated empty SNN graph by %s\n", inputModelFormat.c_str());
            SNN_CHECK_SUCCESS(snnGraph.size() != 0, true);
            return false;
        }
        outputIndex = mModelMaps[modelName]["outputIndex"];
        printf("================== INFO: Finsh converting to SNN nodes ... ==================\n");
        return true;
    }
    bool Model::BuildSNNGraph(bool is_optimization)
    {
        netOpContainer.reserve(interpreter->numOperators);
        std::string name;
        std::shared_ptr<Tensor> tensor;
        for (int i = 0; i < interpreter->numOperators; i++)
        {
            tensor = snnGraph[i];
            name = tensor->GetOpName();
            tensor->SetMainMemory(mainMemory);
            this->backend->BuildOperation(tensor, netOpContainer);
        }
        enableOptimization = is_optimization;
        if (enableOptimization)
        {
            GraphOptimization();
            optMTFDGraph = optGraph;
            optMTFDOpContainer = optOpContainer;
            optMTFDGraphLinks = connection_infos;
            optGraph.clear();
            optOpContainer.clear();
            nodeLength = optMTFDOpContainer.size();
        }
        else
        {
            MTFDGraph = snnGraph;
            MTFDOpContainer = netOpContainer;
            nodeLength = interpreter->numOperators;
        }
        snnGraph.clear();
        netOpContainer.clear();
        return true;
    }

    void Model::GraphOptimization()
    {
        printf("INFO: Start optimizing model graph ...\n");
        int i = 0, j = 0, k = 0;
        std::shared_ptr<Tensor> tensor0, tensor1, tensor2;
        std::shared_ptr<Execution> op;
        std::vector<std::shared_ptr<Tensor>> inputs, outputs;
        optOpContainer.reserve(interpreter->numOperators);
        optGraph.reserve(interpreter->numOperators);
        bool status;
        float *inputData = (float *)malloc(320 * 320 * 3 * sizeof(float));
        string filename = "/aidata/anders/data_collection/okay/demo_test/imgs/frame-000867.jpg";
        cv::Mat frame = cv::imread(filename);
        demo_frame = frame;
        cv::resize(frame, frame, cv::Size(320, 320), 0.5, 0.5, cv::INTER_AREA);
        frame.convertTo(frame, CV_32F, 1.0 / 255.0, 0);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        inputs.emplace_back(snnGraph[0]);
        outputs.emplace_back(snnGraph[0]);
        status = netOpContainer[0]->onInputExecute(inputData, inputs, outputs);
        status = netOpContainer[0]->onInputExecute((float *)frame.datastart, inputs, outputs);

        inputs.clear();
        outputs.clear();
        for (i = 1; i < interpreter->numOperators; i++)
        {
            tensor0 = snnGraph[i];
            op = netOpContainer[i];
            const std::vector<int> &inputIndex = tensor0->inputIndex;
            for (j = 0; j < inputIndex.size(); j++)
            {
                inputs.emplace_back(snnGraph.at(inputIndex[j]));
            }
            outputs.emplace_back(snnGraph.at(i));
            status = op->onExecute(inputs, outputs);
            inputs.clear();
            outputs.clear();
        }
        // float *outputData = op->onConvert(snnGraph[108]);
        // int size = 20 * 20 * 8;
        // FILE *pfile;
        // pfile = fopen("/aidata/anders/data_collection/okay/WF/archives/test/test_data/optimization/original.binary", "wb");
        // fwrite(outputData, 1, size * sizeof(float), pfile);
        // fclose(pfile);
        int counts = 0;
        std::map<int, int> i2fused_op;
        std::vector<std::vector<int>> optimizedConnections;
        optimizedConnections.reserve(interpreter->numOperators - 2);
        for (i = 0; i < interpreter->numOperators; i++)
        {
            std::vector<std::shared_ptr<Tensor>> blk;
            blk.reserve(3);
            tensor0 = snnGraph[i];
            op = netOpContainer[i];
            const std::vector<int> &tensor0_inputIndex = tensor0->inputIndex;
            const std::vector<int> &outputIndex = tensor0->outputIndex;
            // printf("---------------------------\n");
            if (i < interpreter->numOperators - 2)
            {
                tensor1 = snnGraph[i + 1];
                const std::vector<int> &inputIndex = tensor1->inputIndex;
                tensor2 = snnGraph[i + 2];
                const std::vector<int> &kernelShape0 = tensor0->KernelShape();
                const std::vector<int> &kernelShape1 = tensor1->KernelShape();
                const std::vector<int> &kernelShape2 = tensor2->KernelShape();
                const std::vector<int> &outputShape0 = tensor0->OutputShape();
                const std::vector<int> &outputShape1 = tensor1->OutputShape();
                const std::vector<int> &outputShape2 = tensor2->OutputShape();
                if (tensor0->GetOpType() == INPUTDATA)
                {
                    blk.emplace_back(tensor0);
                    blk.emplace_back(tensor1);
                    this->backend->MergedOperators(blk, optOpContainer);
                    i += 1;
                }
                else if ((kernelShape0[2] == 3) && (kernelShape0[3] == 3) && (kernelShape1[3] == 3) && (kernelShape1[3] == 3))
                {
                    blk.emplace_back(tensor0);
                    optOpContainer.emplace_back(op);
                }
                else if ((outputShape0[1] != outputShape1[1]) && (outputShape0[2] != outputShape1[2]))
                {
                    blk.emplace_back(tensor0);
                    optOpContainer.emplace_back(op);
                }
                else if (tensor0->GetOpType() == RESIZE_NEAREST_NEIGHBOR)
                {
                    if (tensor1->GetOpType() == ADD)
                    {
                        bool order = inputIndex[0] > inputIndex[1] ? true : false;
                        if (order)
                        {
                            blk.emplace_back(snnGraph[inputIndex[0]]);
                            blk.emplace_back(snnGraph[inputIndex[1]]);
                        }
                        else
                        {
                            blk.emplace_back(snnGraph[inputIndex[1]]);
                            blk.emplace_back(snnGraph[inputIndex[0]]);
                        }
                        blk.emplace_back(tensor1);
                        this->backend->MergedOperators(blk, optOpContainer);
                        i += 1;
                    }
                    else
                    {
                        blk.emplace_back(tensor0);
                        optOpContainer.emplace_back(op);
                    }
                }
                else if (tensor0->GetOpType() == CONV2D)
                {

                    if (tensor1->GetOpType() == ADD)
                    {
                        bool order = inputIndex[0] > inputIndex[1] ? true : false;
                        if (order)
                        {
                            blk.emplace_back(snnGraph[inputIndex[0]]);
                            blk.emplace_back(snnGraph[inputIndex[1]]);
                        }
                        else
                        {
                            blk.emplace_back(snnGraph[inputIndex[1]]);
                            blk.emplace_back(snnGraph[inputIndex[0]]);
                        }
                        blk.emplace_back(tensor1);
                        this->backend->MergedOperators(blk, optOpContainer);
                        i += 1;
                    }
                    else
                    {
                        blk.emplace_back(tensor0);
                        optOpContainer.emplace_back(op);
                    }
                }
                else if (tensor0->GetOpType() == DEPTHWISECONV2D)
                {

                    if (tensor1->GetOpType() == ADD)
                    {

                        bool order = inputIndex[0] > inputIndex[1] ? true : false;
                        if (order)
                        {
                            blk.emplace_back(snnGraph[inputIndex[0]]);
                            blk.emplace_back(snnGraph[inputIndex[1]]);
                        }
                        else
                        {

                            blk.emplace_back(snnGraph[inputIndex[1]]);
                            blk.emplace_back(snnGraph[inputIndex[0]]);
                        }
                        blk.emplace_back(tensor1);
                        this->backend->MergedOperators(blk, optOpContainer);

                        i += 1;
                    }
                    else if ((tensor1->GetOpType() == DEPTHWISECONV2D) && (tensor2->GetOpType() == DEPTHWISECONV2D))
                    {
                        blk.emplace_back(tensor0);
                        if ((kernelShape1[2] == 1) && (kernelShape1[3] == 1))
                        {
                            blk.emplace_back(tensor1);
                            i += 1;
                        }
                        if ((kernelShape2[2] == 1) && (kernelShape2[3] == 1))
                        {
                            blk.emplace_back(tensor2);
                            i += 1;
                        }
                        this->backend->MergedOperators(blk, optOpContainer);
                    }
                    else if ((tensor1->GetOpType() == CONV2D) && (tensor2->GetOpType() == CONV2D))
                    {

                        blk.emplace_back(tensor0);
                        if ((kernelShape1[2] == 1) && (kernelShape1[3] == 1) && (kernelShape2[2] == 1) && (kernelShape2[3] == 1))
                        {
                            blk.emplace_back(tensor1);
                            blk.emplace_back(tensor2);
                            this->backend->MergedOperators(blk, optOpContainer);
                            i += 2;
                        }
                        if ((kernelShape1[2] == 1) && (kernelShape1[3] == 1) && (kernelShape2[2] == 3) && (kernelShape2[3] == 3))
                        {
                            optOpContainer.emplace_back(op);
                        }
                    }
                    // else if (tensor1->GetOpType() == CONV2D)
                    // {

                    //     blk.emplace_back(tensor0);
                    //     if ((kernelShape1[2] == 1) && (kernelShape1[3] == 1) && (kernelShape1[1] <= 32))
                    //     {
                    //         blk.emplace_back(tensor1);
                    //         this->backend->MergedOperators(blk, optOpContainer);
                    //         i += 1;
                    //     }
                    //     else
                    //     {

                    //         optOpContainer.emplace_back(op);
                    //     }
                    // }
                    else
                    {
                        blk.emplace_back(tensor0);
                        optOpContainer.emplace_back(op);
                    }
                }
                else
                {
                    blk.emplace_back(tensor0);
                    optOpContainer.emplace_back(op);
                }
            }
            else
            {
                blk.emplace_back(tensor0);
                optOpContainer.emplace_back(op);
            }
            optGraph.emplace_back(blk);
            i2fused_op[i] = counts;
            if (counts == 0)
            {
                connection_infos[counts] = {i2fused_op[0]};
            }
            else
            {
                if ((blk.size() == 3) && blk[2]->GetOpType() == ADD)
                {
                    int index = blk[2]->inputIndex[0] < blk[2]->inputIndex[1] ? blk[2]->inputIndex[0] : blk[2]->inputIndex[1];
                    connection_infos[counts] = {i2fused_op[tensor0_inputIndex[0]], i2fused_op[index]};
                }
                else
                {
                    connection_infos[counts] = {i2fused_op[tensor0_inputIndex[0]]};
                }
            }
            counts += 1;
        }
        std::vector<std::shared_ptr<Tensor>> output_tensors{optGraph.at(61)[0], optGraph.at(60)[0], optGraph.at(64)[0], optGraph.at(67)[0]};
        this->backend->PostOperators(output_tensors, optOpContainer);
        optGraph.emplace_back(output_tensors);
        free(inputData);
        printf("INFO: ======================= Finish optimization =======================\n");
        int numInput1, numInput2;
        std::vector<std::shared_ptr<Tensor>> tensors;
        for (i = 0; i < counts; i++)
        {
            op = optOpContainer[i];
            if (i == 0)
            {
                status = op->onInputExecute((float *)frame.datastart, optGraph[connection_infos[i][0]], optGraph[i]);
            }
            else
            {
                if (connection_infos[i].size() == 2)
                {
                    numInput1 = optGraph[connection_infos[i][0]].size();
                    numInput2 = optGraph[connection_infos[i][1]].size();
                    tensors = {optGraph[connection_infos[i][0]][numInput1 - 1], optGraph[connection_infos[i][1]][numInput2 - 1]};
                    status = op->onOptimizedExecute(tensors, optGraph[i]);
                }
                else
                {
                    status = op->onOptimizedExecute(optGraph[connection_infos[i][0]], optGraph[i]);
                }
            }
        }
        // std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> results;
        // exit(1);
        // tensor0 = optGraph.at(64)[0];
        // param_x = op->onConvert(tensor0);
        // tensor0 = optGraph.at(67)[0];
        // trans_x = op->onConvert(tensor0);
        // mpostProcessor->MTFDBaseProcessor(results,
        //                                   1,
        //                                   resizedRatios,
        //                                   cls_x,
        //                                   bbox_x,
        //                                   param_x,
        //                                   trans_x);
        // ShowBbox(demo_frame, results.first);
        // ShowLandmarks(demo_frame, results.second);
        // cv::imwrite("output.jpg", demo_frame);
        // free(cls_x);
        // free(bbox_x);
        // free(param_x);
        // free(trans_x);
    }
    void *Model::InputPreprocess(void *args)
    {
        streaming_t *tabs = (streaming_t *)args;
        bool ret, status;
        int i = 0, batch = bCounts;
        // std::string name;
        std::vector<std::shared_ptr<Tensor>> inputs, outputs;
        inputs.reserve(1);
        outputs.reserve(1);
        std::shared_ptr<Execution> op;
        while (1)
        {
            // auto beg = std::chrono::high_resolution_clock::now();
            ret = tabs->cap->read(frame);
            if (!ret)
            {
                printf("Can't receive frame (stream end?). Exiting ...\n");
                break;
            }
            pthread_mutex_lock(&lock_show);
            demo_frame = frame;
            // status_show = true;
            pthread_cond_signal(&cond_show);
            pthread_mutex_unlock(&lock_show);
            cv::resize(frame, frame, cv::Size(320, 320), 0.5, 0.5, cv::INTER_LINEAR);
            frame.convertTo(frame, CV_32F, 1.0 / 255.0, 0);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            // frame.convertTo(frame, CV_32FC3);
            // memcpy(batchBuffer + i * tabs->bufferSize, (float *)frame.data, tabs->bufferSize * sizeof(float));
            i++;
            batch--;
            if (batch == 0)
            {
                pthread_mutex_lock(&lock_model);
                if (enableOptimization)
                {
                    status = optMTFDOpContainer[0]->onInputExecute((float *)frame.datastart, optMTFDGraph[optMTFDGraphLinks[0][0]], optMTFDGraph[0]);
                }
                else
                {
                    inputs.emplace_back(MTFDGraph[0]);
                    outputs.emplace_back(MTFDGraph[0]);
                    status = MTFDOpContainer[0]->onInputExecute((float *)frame.datastart, inputs, outputs);
                    inputs.clear();
                    outputs.clear();
                }
                status_model = true;
                pthread_cond_signal(&cond_model);
                pthread_mutex_unlock(&lock_model);
                batch = bCounts;
                i = 0;
            }
            // cv::imshow("DEMO", demo_frame);
            // if (cv::waitKey(1) == 'q')
            // {
            //     exitSignal = true;
            //     break;
            // }
            if (exitSignal)
                break;
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
            // std::cout << "Elapsed Time: " << duration.count() * 1e-6 << std::endl;
        }
        return nullptr;
    }
    void *Model::OperatorInference(void *args)
    {
        streaming_t *tabs = (streaming_t *)args;
        std::shared_ptr<Tensor> tensor;
        std::vector<std::shared_ptr<Tensor>> inputs, outputs, tensors;
        std::shared_ptr<Execution> op;
        std::string name;
        int i, j, numInput1, numInput2;
        bool status;
        inputs.reserve(3);
        outputs.reserve(3);
        float *cls_x, *bbox_x, *param_x, *trans_x;
        std::unique_ptr<PostProcessor> postProcessor = std::make_unique<PostProcessor>(thread_model_cfg);
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> results;
        while (1)
        {
            // auto beg = std::chrono::high_resolution_clock::now();
            pthread_mutex_lock(&lock_model);
            while (!status_model)
                pthread_cond_wait(&cond_model, &lock_model);

            if (enableOptimization)
            {

                for (i = 1; i < nodeLength; i++)
                {
                    op = optMTFDOpContainer[i];
                    if (optMTFDGraphLinks[i].size() == 2)
                    {
                        numInput1 = optMTFDGraph[optMTFDGraphLinks[i][0]].size();
                        numInput2 = optMTFDGraph[optMTFDGraphLinks[i][1]].size();
                        inputs = {optMTFDGraph[optMTFDGraphLinks[i][0]][numInput1 - 1], optMTFDGraph[optMTFDGraphLinks[i][1]][numInput2 - 1]};
                        outputs = optMTFDGraph[i];
                        status = op->onOptimizedExecute(inputs, outputs);
                    }
                    else
                    {
                        status = op->onOptimizedExecute(optMTFDGraph[optMTFDGraphLinks[i][0]], optMTFDGraph[i]);
                    }
                }
                tensor = optMTFDGraph.at(60)[0];
                bbox_x = op->onConvert(tensor);
                tensor = optMTFDGraph.at(61)[0];
                cls_x = op->onConvert(tensor);
                tensor = optMTFDGraph.at(64)[0];
                param_x = op->onConvert(tensor);
                tensor = optMTFDGraph.at(67)[0];
                trans_x = op->onConvert(tensor);
            }
            else
            {
                for (i = 1; i < nodeLength - 1; i++)
                {
                    tensor = MTFDGraph.at(i);
                    // name = tensor->GetOpName();
                    // std::cout << " ------------------------ Node op index: ------------------------  " << i << std::endl;
                    // std::cout << name << std::endl;
                    const std::vector<int> &inputIndex = tensor->inputIndex;
                    op = MTFDOpContainer[i];
                    for (j = 0; j < inputIndex.size(); j++)
                    {
                        inputs.emplace_back(MTFDGraph.at(inputIndex[j]));
                    }
                    outputs.emplace_back(MTFDGraph.at(i));
                    status = op->onExecute(inputs, outputs);
                    inputs.clear();
                    outputs.clear();
                }

                tensor = MTFDGraph.at(outputIndex[0]);
                bbox_x = op->onConvert(tensor);
                tensor = MTFDGraph.at(outputIndex[1]);
                cls_x = op->onConvert(tensor);
                tensor = MTFDGraph.at(outputIndex[2]);
                param_x = op->onConvert(tensor);
                tensor = MTFDGraph.at(outputIndex[3]);
                trans_x = op->onConvert(tensor);
            }
            // postProcessor->MTFDBaseProcessor(results,
            //                                  1,
            //                                  resizedRatios,
            //                                  cls_x,
            //                                  bbox_x,
            //                                  param_x,
            //                                  trans_x);
            status_model = false;
            // ShowBbox(demo_frame, results.first);
            // ShowLandmarks(demo_frame, results.second);
            // cv::imshow("DEMO", demo_frame);
            pthread_mutex_unlock(&lock_model);
            // if (cv::waitKey(1) == 'q')
            // {
            //     exitSignal = true;
            //     break;
            // }
            free(cls_x);
            free(bbox_x);
            free(param_x);
            free(trans_x);
            // results.first.clear();
            // results.second.clear();
            // std::vector<std::vector<float>> n_bboxes = results.first;
            // std::vector<std::vector<float>> n_landmarks = results.second[0];
            // for (i = 0; i < 68; i++)
            // {
            //     std::cout << "(" << n_landmarks[i][0] << ", " << n_landmarks[i][1] << ")" << std::endl;
            // }
            // printf("%ld\n", n_landmarks.size());
            // printf("%ld\n", n_bboxes.size());

            // std::cout << "(" << n_bboxes[0][0] << ", " << n_bboxes[0][1] << ", " << n_bboxes[0][2] << ", " << n_bboxes[0][3]
            //           << ")" << std::endl;
            // exit(1);
            // if (exitSignal)
            //     break;
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
            // std::cout << "Elapsed Time: " << duration.count() * 1e-6 << std::endl;
        }
        return nullptr;
    }
    void *Model::Display(void *args)
    {
        while (1)
        {
            // auto beg = std::chrono::high_resolution_clock::now();
            pthread_mutex_lock(&lock_show);
            while (!status_show)
                pthread_cond_wait(&cond_show, &lock_show);
            status_show = false;
            cv::imshow("DEMO", demo_frame);
            pthread_mutex_unlock(&lock_show);
            if (cv::waitKey(1) == 'q')
            {
                exitSignal = true;
                break;
            }
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
            // std::cout << "Elapsed Time: " << duration.count() * 1e-6 << std::endl;
        }
        return nullptr;
    }
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> Model::Inference(cv::VideoCapture &cap, float *resizedRatios)
    {

        bool ret;
        cv::Mat frame, demo_frame;
        int i, j;
        bool status;
        std::shared_ptr<Execution> op;
        std::vector<int> input_idx, output_idx;
        std::vector<std::shared_ptr<Tensor>> inputs, outputs;
        inputs.reserve(3);
        outputs.reserve(3);
        std::shared_ptr<Tensor> tensor;
        std::string name;
        int counts = 0;
        while (1)
        {
            if (counts == 0)
            {
                break;
            }
            ret = cap.read(frame);
            demo_frame = frame;
            float *input_data = (float *)frame.datastart;
            for (i = 0; i < interpreter->numOperators - 1; i++)
            {
                // std::cout << " ------------------------ Node op index: ------------------------  " << i << std::endl;
                tensor = snnGraph.at(i);
                // name = tensor->GetOpName();
                // std::cout << name << std::endl;
                const std::vector<int> &inputIndex = tensor->inputIndex;
                op = netOpContainer[i];
                if (i == 0)
                {
                    inputs.emplace_back(snnGraph.at(0));
                    outputs.emplace_back(snnGraph.at(0));
                    status = op->onInputExecute(input_data, inputs, outputs);
                }
                else
                {
                    for (j = 0; j < inputIndex.size(); j++)
                    {
                        inputs.emplace_back(snnGraph.at(inputIndex[j]));
                    }
                    outputs.emplace_back(snnGraph.at(i + 1));
                    status = op->onExecute(inputs, outputs);
                }

                inputs.clear();
                outputs.clear();
            }
        }

        float *cls_x, *bbox_x, *param_x, *trans_x;
        std::vector<int> &inputIndex = mModelMaps[modelName]["inputIndex"];
        std::vector<int> &outputIndex = mModelMaps[modelName]["outputIndex"];
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> results;
        for (i = 0; i < outputIndex.size() / 4; i++)
        {
            if (i == 0 || i == 2)
                continue;
            if (i == 2)
            {
                tensor = snnGraph.at(outputIndex[4 * i]);
                param_x = op->onConvert(tensor);
                tensor = snnGraph.at(outputIndex[4 * i + 1]);
                trans_x = op->onConvert(tensor);
                tensor = snnGraph.at(outputIndex[4 * i + 2]);
                bbox_x = op->onConvert(tensor);
                tensor = snnGraph.at(outputIndex[4 * i + 3]);
                cls_x = op->onConvert(tensor);
            }
            else
            {
                tensor = snnGraph.at(outputIndex[4 * i]);
                bbox_x = op->onConvert(tensor);
                tensor = snnGraph.at(outputIndex[4 * i + 1]);
                cls_x = op->onConvert(tensor);
                tensor = snnGraph.at(outputIndex[4 * i + 2]);
                param_x = op->onConvert(tensor);
                tensor = snnGraph.at(outputIndex[4 * i + 3]);
                trans_x = op->onConvert(tensor);
            }
            mpostProcessor->MTFDBaseProcessor(results,
                                              i,
                                              resizedRatios,
                                              cls_x,
                                              bbox_x,
                                              param_x,
                                              trans_x);
            free(cls_x);
            free(bbox_x);
            free(param_x);
            free(trans_x);
        }
        return results;
    }
} // namespace SNN
