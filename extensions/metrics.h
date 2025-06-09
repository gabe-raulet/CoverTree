#ifndef METRICS_H_
#define METRICS_H_

#include <cmath>

struct EuclideanDistance
{
    float operator()(const float *p, const float *q, int d) const
    {
        float val = 0;

        for (int i = 0; i < d; ++i)
        {
            float delta = p[i] - q[i];
            val += delta * delta;
        }

        return std::sqrt(val);
    }
};

struct ManhattanDistance
{
    float operator()(const float *p, const float *q, int d) const
    {
        float val = 0;

        for (int i = 0; i < d; ++i)
        {
            val += std::abs(p[i] - q[i]);
        }

        return val;
    }
};

struct CosineDistance
{
    float operator()(const float *p, const float *q, int d) const
    {
        float val = 0;

        for (int i = 0; i < d; ++i)
            val += p[i]*q[i];

        return 1.0 - val;
    }
};

struct AngularDistance
{
    float operator()(const float *p, const float *q, int d) const
    {
        float pq = 0, pp = 0, qq = 0;

        for (int i = 0; i < d; ++i)
        {
            pq += p[i]*q[i];
            pp += p[i]*p[i];
            qq += q[i]*q[i];
        }


        if (pp*qq == 0)
            return 0;

        float val = pq / std::sqrt(pp*qq);
        return acos(val) / M_PI;
    }
};

struct ChebyshevDistance
{
    float operator()(const float *p, const float *q, int d) const
    {
        float val = 0;

        for (int i = 0; i < d; ++i)
        {
            val = std::max(val, std::abs(p[i] - q[i]));
        }

        return val;
    }
};

#endif
