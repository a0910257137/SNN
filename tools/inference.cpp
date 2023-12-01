
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
    // string filename = "/aidata/anders/data_collection/okay/demo_test/imgs/frame-000867.jpg";
    // float *input_data = (float *)malloc(320 * 320 * 3 * sizeof(float));
    float *input_data = (float *)malloc(720 * 1280 * 3 * sizeof(float));
    float *resizedRatios = (float *)malloc(2 * sizeof(float));
    // readImage(filename, input_data, resizedRatios);
    pipeline.BuildSNNGraph();
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cout << "Cannot open camera\n";
        return 1;
    }
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>> results;
    float elapsed_time;
    cv::Mat frame, demo_frame;
    bool ret;
    float FPS_f;
    string FPS_s;
    while (true)
    {
        ret = cap.read(frame);
        demo_frame = frame;
        if (!ret)
        {
            cout << "Can't receive frame (stream end?). Exiting ...\n";
            break;
        }
        clock_gettime(CLOCK_REALTIME, &ivk_s);
        readVideo(frame, input_data, resizedRatios, true);
        results = pipeline.Inference((float *)frame.datastart, resizedRatios);
        clock_gettime(CLOCK_REALTIME, &ivk_e);
        ShowBbox(demo_frame, results.first);
        ShowLandmarks(demo_frame, results.second);
        elapsed_time = diff_ms(&ivk_e, &ivk_s);
        FPS_f = (1000.0f) / elapsed_time;
        FPS_s = FormatZeros(FPS_f);
        FPS_s = "FPS: " + FPS_s;
        ShowText(demo_frame, FPS_s);
        printf("Elaspe time in inference: %3.3f\n", elapsed_time);
        // cv::imshow("DEMO", demo_frame);
        // if (cv::waitKey(1) == 'q')
        // {
        //     break;
        // }
    }
    free(input_data);
    free(resizedRatios);
    return 1;
}