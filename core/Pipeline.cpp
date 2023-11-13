#include "Pipeline.h"
namespace SNN
{
    Pipeline::Pipeline(ModelConfig &model_cfg, BackendConfig &backend_cfg) : Model(model_cfg)
    {
        this->backend = std::make_shared<Backend>(backend_cfg);
        this->GetSNNGraph();
    }
    Pipeline::~Pipeline()
    {
        this->backend.reset();
    }
} // namespace  SNN
