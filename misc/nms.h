#ifndef NMS_H__
#define NMS_H__
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

enum PointInRectangle
{
    XMIN = 0,
    YMIN = 1,
    XMAX = 2,
    YMAX = 3,
    SCRORE = 4,
    CLASS = 5

};
std::pair<int, std::vector<float>> NMS(std::vector<std::vector<float>> &, const float &);
std::vector<float> GetPointFromRect(std::vector<std::vector<float>> &, const PointInRectangle &);
std::vector<float> ComputeArea(const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &);
template <typename T>
std::vector<int> argsort(const std::vector<T> &);
std::vector<int> RemoveLast(const std::vector<int> &);

std::vector<float> CopyByIndexes(const std::vector<float> &, const std::vector<int> &);

std::vector<float> Maximum(const float &, const std::vector<float> &);

std::vector<float> Minimum(const float &, const std::vector<float> &);

std::vector<float> Subtract(const std::vector<float> &, const std::vector<float> &);

std::vector<float> Multiply(const std::vector<float> &, const std::vector<float> &);

std::vector<float> Divide(const std::vector<float> &, const std::vector<float> &);

std::vector<int> WhereLarger(const std::vector<float> &, const float);

std::vector<int> RemoveByIndexes(const std::vector<int> &, const std::vector<int> &);

std::pair<int, std::vector<float>> PickBboxes(std::vector<std::vector<float>> &, std::vector<int>);

std::vector<float> SelectLargeBox(std::vector<float> &);
#endif