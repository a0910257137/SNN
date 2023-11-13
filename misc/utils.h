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

void readImage(std::string &filename, float *output);
static inline int argmax(std::vector<float> *);
float diff_ms(struct timespec *t1, struct timespec *t0);
void print(int);
#endif // UTILS_H__
