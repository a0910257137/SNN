#ifndef PIPELINE_H
#define PIPELINE_H
#include <string>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <memory>
#include <stdlib.h>
#include <stdio.h>
#include "include/SNN/common.h"
#include "include/SNN/api_types.h"
#include "include/SNN/SNNDefine.h"
#include "include/SNN/SNNDefine.h"
#include "BaseProctocol.h"
#include "Interpreter.h"

namespace SNN
{
    /*
     * Pipeline contains miltiple models
     * One model contains many operators.
     * */
    template <typename T>
    struct array_deleter
    {
        void operator()(T const *p)
        {
            delete[] p;
        }
    };
    class Pipeline : public BaseProtocol
    {
    public:
        Pipeline(std::string model_path, DeviceType device_type);
        ~Pipeline();

    public:
        bool GetSNNGraph();
        bool BuildSNNGraph();
        bool AllocMemory(bool firstMalloc);
        bool Execute();
        void Run();

    private:
        bool mAllocInput;
        bool mOutputStatic;
        bool msupportModel = false;
        bool firstMalloc = true;
        std::string inputModelFormat;
        std::shared_ptr<bool[]> firstMallocs;
        std::shared_ptr<NodeFactory> nodefactory;
        std::unique_ptr<Interpreter> interpreter;
        std::vector<SNNNode> snnGraph;
    };
}
#endif // PIPELINE_H