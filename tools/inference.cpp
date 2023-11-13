
#include "core/Pipeline.h"
#include "misc/utils.h"
using namespace std;
timespec ivk_s, ivk_e;
int main(int argc, char **argv)
{
    BackendConfig backend_cfg;
    ModelConfig model_cfg;
    SNN::Pipeline pipeline(model_cfg, backend_cfg);
    string filename = "/aidata/anders/data_collection/okay/demo_test/imgs/frame-000867.jpg";
    float *input_data = (float *)malloc(320 * 320 * 3 * sizeof(float));
    readImage(filename, input_data);
    pipeline.BuildSNNGraph();
    while (1)
    {
        clock_gettime(CLOCK_REALTIME, &ivk_s);
        pipeline.Inference(input_data);
        clock_gettime(CLOCK_REALTIME, &ivk_e);
        float diff = diff_ms(&ivk_e, &ivk_s);
        printf("Elaspe time in inference: %3.3f\n", diff);
        // std::cout << "Elaspe time in inference:  " << diff << std::endl;
    }
    // clock_gettime(CLOCK_REALTIME, &ivk_s);
    // pipeline.Inference(input_data);
    // clock_gettime(CLOCK_REALTIME, &ivk_e);
    // float diff = diff_ms(&ivk_e, &ivk_s);
    // std::cout << "Elaspe time in inference:  " << diff << std::endl;
    free(input_data);
    return 1;
}