#include "DepthwiseConvExecution.h"
namespace SNN
{
    DepthwiseConvExecution::DepthwiseConvExecution(Tensor *inputs)
    {

        printf("%d\n", inputs->op_type);
    }
    DepthwiseConvExecution::~DepthwiseConvExecution()
    {
    }
}