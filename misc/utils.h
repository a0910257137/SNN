#ifndef UTILS_H__
#define UTILS_H__
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <dirent.h>
void readVideo(cv::Mat src, float *dts, float *resizedRatios);
void readImage(std::string &filename, float *output, float *resizedRatios);
static inline int argmax(std::vector<float> *);
float diff_ms(struct timespec *t1, struct timespec *t0);
void print(int);
std::string base_name(std::string const &path);
std::string remove_extension(std::string const &filename);
std::tuple<int, float *> _readBinary(std::string &file_name, FILE *ptr);
void GetWeights(std::string &path, std::map<std::string, std::tuple<int, float *>> &headWeights);
std::string FormatZeros(float value);
void ShowBbox(cv::Mat &frame, std::vector<std::vector<float>> &nbbox);
void ShowLandmarks(cv::Mat &frame, std::vector<std::vector<std::vector<float>>> &nlnmks);
void PutText(cv::Mat &frame, const std::string &text, int t_h_o, cv::Scalar &clc, int padding);
void ShowText(cv::Mat &frame, std::string &text);
#endif // UTILS_H__
