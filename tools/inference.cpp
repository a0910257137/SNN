
#include "core/Pipeline.h"
#include "misc/utils.h"
using namespace std;
timespec ivk_s, ivk_e;
int main(int argc, char **argv)
{
    BackendConfig backend_cfg;
    ModelConfig model_cfg;
    SNN::Pipeline pipeline(model_cfg, backend_cfg);
    string video_path = "/aidata/anders/landmarks/demo_video/2021_12_24/[drive]anders_2.MP4";
    string filename = "/aidata/anders/data_collection/okay/demo_test/imgs/frame-000867.jpg";
    float *input_data = (float *)malloc(320 * 320 * 3 * sizeof(float));
    float *resizedRatios = (float *)malloc(2 * sizeof(float));
    readImage(filename, input_data, resizedRatios);
    pipeline.BuildSNNGraph();
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cout << "Cannot open camera\n";
        return 1;
    }
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> results;
    cv::Mat frame;
    bool ret = true;
    float elapsed_time, FPS_f;
    string FPS_s;
    while (true)
    {
        ret = cap.read(frame);
        if (!ret)
        {
            cout << "Can't receive frame (stream end?). Exiting ...\n";
            break;
        }
        readVideo(frame, input_data, resizedRatios);
        clock_gettime(CLOCK_REALTIME, &ivk_s);
        results = pipeline.Inference(input_data, resizedRatios);
        clock_gettime(CLOCK_REALTIME, &ivk_e);
        ShowBbox(frame, results.first);
        ShowLandmarks(frame, results.second);
        elapsed_time = diff_ms(&ivk_e, &ivk_s);
        FPS_f = (1000.0f) / elapsed_time;
        FPS_s = FormatZeros(FPS_f);
        FPS_s = "FPS: " + FPS_s;
        ShowText(frame, FPS_s);
        // printf("Elaspe time in inference: %3.3f\n", elapsed_time);
        // std::cout << "Elaspe time in inference:  " << elapsed_time << std::endl;
        cv::imshow("DEMO", frame);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }
    free(input_data);
    free(resizedRatios);
    return 1;
}