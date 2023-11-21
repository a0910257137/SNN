#include "nms.h"

std::pair<int, std::vector<float>> NMS(std::vector<std::vector<float>> &boxes, const float &thres)
{
    /*
        With non-max suppression, we could get the maximum IoU boudning box
    */
    const std::vector<float> x1 = GetPointFromRect(boxes, XMIN);
    const std::vector<float> y1 = GetPointFromRect(boxes, YMIN);
    const std::vector<float> x2 = GetPointFromRect(boxes, XMAX);
    const std::vector<float> y2 = GetPointFromRect(boxes, YMAX);
    const std::vector<float> sc = GetPointFromRect(boxes, SCRORE);
    std::vector<float> area = ComputeArea(x1, y1, x2, y2);
    std::vector<int> idxs = argsort(sc);

    int i, last;
    std::vector<int> pick;
    while (idxs.size() > 0)
    {
        last = idxs.size() - 1;
        i = idxs[last];
        pick.push_back(i);
        auto idxsWoLast = RemoveLast(idxs);
        auto xx1 = Maximum(x1[i], CopyByIndexes(x1, idxsWoLast));
        auto yy1 = Maximum(y1[i], CopyByIndexes(y1, idxsWoLast));
        auto xx2 = Minimum(x2[i], CopyByIndexes(x2, idxsWoLast));
        auto yy2 = Minimum(y2[i], CopyByIndexes(y2, idxsWoLast));
        // compute the width and height of the bounding box
        auto w = Maximum(0, Subtract(xx2, xx1));
        auto h = Maximum(0, Subtract(yy2, yy1));
        auto overlap = Divide(Multiply(w, h), CopyByIndexes(area, idxsWoLast));
        auto deleteIdxs = WhereLarger(overlap, thres);
        deleteIdxs.push_back(last);
        idxs = RemoveByIndexes(idxs, deleteIdxs);
    }
    std::pair<int, std::vector<float>> outputBboxes;
    if (pick.size() > 0)
    {
        outputBboxes = PickBboxes(boxes, pick);
    }
    return outputBboxes;
}

std::vector<float> SelectLargeBox(std::vector<float> &boxVec)
{

    std::vector<float> areaVec;
    for (int i = 0; i < boxVec.size() % 4; i++)
    {
        float area = (boxVec[i + 2] - boxVec[i] + 1) *
                     (boxVec[i + 3] - boxVec[i + 1] + 1);
        areaVec.push_back(area);
    }
    std::vector<int> idxs = argsort(areaVec);

    int sidx = idxs[-1];

    std::vector<float> resultVect{boxVec[sidx],
                                  boxVec[sidx + 1],
                                  boxVec[sidx + 2],
                                  boxVec[sidx + 3]};
    return resultVect;
}
std::vector<float> GetPointFromRect(std::vector<std::vector<float>> &boxes, const PointInRectangle &pos)
{
    int numDetSize = boxes.size();
    std::vector<float> points(numDetSize, 0);
    for (int i = 0; i < numDetSize; i++)
    {
        if (boxes[i].empty())
            break;
        points[i] = boxes[i][pos];
        // std::cout << points[i] << std::endl;
    }
    return points;
}

std::vector<float> ComputeArea(const std::vector<float> &x1,
                               const std::vector<float> &y1,
                               const std::vector<float> &x2,
                               const std::vector<float> &y2)
{

    int numDetSize = x1.size();
    float area;
    std::vector<float> areas(numDetSize, 0);
    for (int i = 0; i < numDetSize; ++i)
    {
        area = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
        // std::cout << area << std::endl;
        areas[i] = area;
    }
    return areas;
}

template <typename T>
std::vector<int> argsort(const std::vector<T> &v)
{
    // sort values and its index
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](int l1, int l2)
              {
                  return v[l1] < v[l2];
              });
    return idx;
}

std::vector<int> RemoveLast(const std::vector<int> &vect)
{
    std::vector<int> resultVec = vect;
    if (resultVec.empty())
        return resultVec;
    resultVec.erase(resultVec.end() - 1);
    return resultVec;
}
std::vector<float> CopyByIndexes(const std::vector<float> &vect, const std::vector<int> &idxs)
{
    std::vector<float> resultVec;
    for (const auto &p : idxs)
        resultVec.push_back(vect[p]);
    return resultVec;
}

std::vector<float> Maximum(const float &val, const std::vector<float> &vect)
{
    std::vector<float> maxVect = vect;
    auto len = vect.size();
    for (decltype(len) idx = 0; idx < len; ++idx)
    {
        if (vect[idx] < val)
            maxVect[idx] = val;
    }
    return maxVect;
}

std::vector<float> Minimum(const float &val, const std::vector<float> &vect)
{
    std::vector<float> minVect = vect;
    auto len = vect.size();
    for (decltype(len) idx = 0; idx < len; ++idx)
    {
        if (vect[idx] > val)
            minVect[idx] = val;
    }
    return minVect;
}
std::vector<float> Substract(const std::vector<float> &vect1, const std::vector<float> &vect2)
{
    std::vector<float> resultVec;
    auto len = vect1.size();
    for (decltype(len) idx = 0; idx < len; ++idx)
        resultVec.push_back(vect1[idx] - vect2[idx] + 1);
    return resultVec;
}

std::vector<float> Multiply(const std::vector<float> &vec1,
                            const std::vector<float> &vec2)
{
    std::vector<float> resultVec;
    auto len = vec1.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        resultVec.push_back(vec1[idx] * vec2[idx]);

    return resultVec;
}
std::vector<float> Divide(const std::vector<float> &vec1,
                          const std::vector<float> &vec2)
{
    std::vector<float> resultVec;
    auto len = vec1.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        resultVec.push_back(vec1[idx] / vec2[idx]);

    return resultVec;
}
std::vector<float> Subtract(const std::vector<float> &vec1,
                            const std::vector<float> &vec2)
{
    std::vector<float> result;
    auto len = vec1.size();

    for (decltype(len) idx = 0; idx < len; ++idx)
        result.push_back(vec1[idx] - vec2[idx] + 1);

    return result;
}

std::vector<int> WhereLarger(const std::vector<float> &vect, const float val)
{
    std::vector<int> resultVec;
    auto len = vect.size();
    for (decltype(len) idx = 0; idx < len; ++idx)
    {
        if (vect[idx] > val)
            resultVec.push_back(idx);
    }
    return resultVec;
}

std::vector<int> RemoveByIndexes(const std::vector<int> &vec,
                                 const std::vector<int> &idxs)
{
    auto resultVec = vec;
    auto offset = 0;
    for (const auto &idx : idxs)
    {

        resultVec.erase(resultVec.begin() + idx + offset);
        --offset;
    }
    return resultVec;
}

std::pair<int, std::vector<float>> PickBboxes(std::vector<std::vector<float>> &boxes, std::vector<int> pick)
{
    int i;
    float area;
    std::vector<float> outputBboxes(4, 0);
    std::pair<int, std::vector<float>> results;
    if (~pick.empty())
    {
        if (pick.size() == 1)
        {
            for (i = 0; i < 4; i++)
                outputBboxes[i] = boxes[pick[0]][i];
            results = std::make_pair(pick[0], outputBboxes);
            return results;
        }

        else
        {
            std::vector<float> areaVec;
            for (i = 0; i < pick.size(); i++)
            {
                area = ((boxes)[pick[i]][2] - (boxes)[pick[i]][0] + 1) *
                       ((boxes)[pick[i]][3] - (boxes)[pick[i]][1] + 1);
                areaVec.push_back(area);
            }
            std::vector<int> idxs = argsort(areaVec);
            int sidx = idxs[idxs.size()];
            for (i = 0; i < 4; i++)
                outputBboxes[i] = boxes[pick[sidx]][i];
            results = std::make_pair(pick[sidx], outputBboxes);
            return results;
        }
    }
}
