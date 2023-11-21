#include "utils.h"

void readVideo(cv::Mat src, float *dts, float *resizedRatios)
{
    // cv::Mat frame = cv::imread(filename);
    resizedRatios[1] = (float)src.rows / 320.0f;
    resizedRatios[0] = (float)src.cols / 320.0f;
    cv::resize(src, src, cv::Size(320, 320), 0.5, 0.5, cv::INTER_AREA);
    cv::cvtColor(src, src, CV_BGR2RGB);
    src.convertTo(src, CV_32F, 1.0 / 255, 0);
    int bytes = src.cols * src.rows * 3 * sizeof(float);
    memcpy(dts, (float *)src.data, bytes);
}

void readImage(std::string &filename, float *output, float *resizedRatios)
{
    cv::Mat frame = cv::imread(filename);
    resizedRatios[1] = (float)frame.rows / 320.0f;
    resizedRatios[0] = (float)frame.cols / 320.0f;
    cv::resize(frame, frame, cv::Size(320, 320), 0.5, 0.5, cv::INTER_AREA);
    cv::cvtColor(frame, frame, CV_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255, 0);
    int bytes = frame.cols * frame.rows * 3 * sizeof(float);
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

std::string base_name(std::string const &path)
{
    return path.substr(path.find_last_of("/") + 1);
}
std::string remove_extension(std::string const &filename)
{
    typename std::string::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != std::string::npos ? filename.substr(0, p) : filename;
}

std::tuple<int, float *> _readBinary(std::string &file_name, FILE *ptr)
{
    std::tuple<int, float *> dtaa;
    ptr = fopen(file_name.c_str(), "rb");
    fseek(ptr, 0, SEEK_END);
    int size = ftell(ptr);
    float *buffer = (float *)malloc(size);
    fseek(ptr, 0, SEEK_SET);
    fread(buffer, size, 1, ptr);
    dtaa = std::make_tuple(size / sizeof(float), buffer);
    return dtaa;
}
void GetWeights(std::string &path, std::map<std::string, std::tuple<int, float *>> &headWeights)
{
    DIR *dir;
    struct dirent *ent;
    std::string b = ".", bb = "..";
    std::string file_name, file_path;
    std::tuple<int, float *> data;
    FILE *ptr;
    if ((dir = opendir(path.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            file_name = ent->d_name;
            if (file_name == b || file_name == bb)
                continue;
            file_path = path + "/" + file_name;
            data = _readBinary(file_path, ptr);
            file_name = remove_extension(base_name(file_name));
            headWeights[file_name] = data;
        }
    }
}

std::string FormatZeros(float value)
{
    // Print value to a string
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    std::string str = ss.str();
    // Ensure that there is a decimal point somewhere (there should be)
    if (str.find('.') != std::string::npos)
    {
        // Remove trailing zeroes

        str = str.substr(0, str.find_last_not_of('0') + 1);
        // If the decimal point is now the last character, remove that as well
        if (str.find('.') == str.size() - 1)
        {
            str = str.substr(0, str.size() - 1);
        }
    }
    return str;
}
void ShowBbox(cv::Mat &frame, std::vector<std::vector<float>> &nbbox)
{
    int i, j;
    for (i = 0; i < nbbox.size(); i++)
    {
        cv::Point tl = cv::Point(nbbox[i][0], nbbox[i][1]);
        cv::Point br = cv::Point(nbbox[i][2], nbbox[i][3]);
        cv::rectangle(frame, cv::Rect(tl, br), cv::Scalar(0, 255, 0), 4, 1, 0);
    }
}
void ShowLandmarks(cv::Mat &frame, std::vector<std::vector<std::vector<float>>> &nlnmks)
{
    std::vector<std::vector<float>> lnmks;
    cv::Point kp;
    for (int i = 0; i < nlnmks.size(); i++)
    {
        lnmks = nlnmks[i];

        for (int j = 0; j < lnmks.size() - 1; j++)
        {
            kp = cv::Point(lnmks[j][0], lnmks[j][1]);
            // std::cout << lnmks.size() << std::endl;
            cv::circle(frame, kp, 4, cv::Scalar(0, 255, 0), -1);
        }
    }
}
void PutText(cv::Mat &frame, const std::string &text, int t_h_o, cv::Scalar &clc, int padding)
{
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0);
    t_h_o = textSize.height + padding;
    cv::putText(
        frame, text, cv::Point(5, t_h_o), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, clc, 2, cv::LINE_AA);
}
void ShowText(cv::Mat &frame, std::string &text)
{
    int t_h_o = 0;
    int base_padding = 11;
    cv::Scalar eye_clc = cv::Scalar(255, 0, 255), pose_clc = cv::Scalar(255, 0, 255);
    cv::Scalar driver_clc = cv::Scalar(0, 255, 255);
    PutText(frame, text, 50, driver_clc, base_padding * 2);
}