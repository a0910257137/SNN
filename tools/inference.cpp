
#include "core/Pipeline.h"
#include "misc/utils.h"
using namespace std;
timespec ivk_s, ivk_e;
int main(int argc, char **argv)
{
    BackendConfig backend_cfg;
    ModelConfig model_cfg;
    SNN::Pipeline pipeline(model_cfg, backend_cfg);
    float elapsed_time;
    pipeline.BuildSNNGraph(true);
    // pipeline.GraphOptimization();
    string video_path = "/aidata/anders/landmarks/demo_video/2021_12_24/[drive]anders_2.MP4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cout << "Cannot open camera\n";
        return 1;
    }
    bool ret;
    float FPS_f;
    string FPS_s;
    streaming_t inputTab;
    pthread_t threads[3];
    inputTab.cap = &cap;
    inputTab.bufferSize = 320 * 320 * 3;
    if (pthread_create(&threads[0], NULL, pipeline.InputPreprocess, &inputTab) != 0)
    {
        fprintf(stderr, "pthread_create failed!\n");
        exit(1);
    }
    if (pthread_create(&threads[1], NULL, pipeline.OperatorInference, &inputTab) != 0)
    {
        fprintf(stderr, "pthread_create failed!\n");
        exit(1);
    }
    // if (pthread_create(&threads[2], NULL, pipeline.Display, &inputTab) != 0)
    // {
    //     fprintf(stderr, "pthread_create failed!\n");
    //     exit(1);
    // }
    if (pthread_join(threads[0], nullptr) != 0 and pthread_join(threads[1], nullptr) != 0)
    {
        fprintf(stderr, "pthread_join failed!\n");
        exit(1);
    }
    return 1;
}