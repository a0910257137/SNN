#include "utils.h"
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