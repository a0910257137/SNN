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
#include "include/SNN/SNNDefine.h"
#include "include/SNN/Tensor.h"
#include "NonCopyable.h"
#include "Interpreter.h"
#include "Model.h"
#include "backend/opencl/core/OpenCLBackend.h"
#include "backend/cpu/CPUBackend.h"
namespace SNN
{
    class Pipeline : public Model
    {

    public:
        Pipeline(ModelConfig &model_cfg, BackendConfig &backend_cfg);
        ~Pipeline();
    };
}
#endif // PIPELINE_H