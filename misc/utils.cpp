#include "utils.h"

void readImage(std::string &filename, float *output)
{
    cv::Mat frame = cv::imread(filename);
    cv::resize(frame, frame, cv::Size(320, 320));
    cv::cvtColor(frame, frame, CV_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255, 0);
    int bytes = (frame.cols) * frame.rows * 3 * sizeof(float);
    memcpy(output, (float *)frame.data, bytes);
}
void print(int n)
{
    for (int i = 0; i < n; ++i)
    {
        printf("-");
    }
    printf("\n");
}

static inline int argmax(std::vector<float> *input)
{
    std::vector<float>::iterator iter = input->begin();
    float value = 0;
    int index, counts;
    while (iter != input->end())
    {
        if (value > *iter)
        {
            value = *iter;
            index = counts;
        }
        counts++;
        iter++;

        if (value > *iter)
        {
            value = *iter;
            index = counts;
        }
        counts++;
        iter++;
    }

    return index;
}
float diff_ms(struct timespec *t1, struct timespec *t0)
{
    struct timespec d = {
        .tv_sec = t1->tv_sec - t0->tv_sec,
        .tv_nsec = t1->tv_nsec - t0->tv_nsec};
    if (d.tv_nsec < 0)
    {
        d.tv_nsec += 1000000000;
        d.tv_sec--;
    }
    return (double)(d.tv_nsec / 1000000);
}