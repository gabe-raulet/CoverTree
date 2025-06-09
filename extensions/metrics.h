#ifndef METRICS_H_
#define METRICS_H_

struct EuclideanDistance
{
    float operator()(const float *p, const float *q, int d)
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
    float operator()(const float *p, const float *q, int d)
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
    float operator()(const float *p, const float *q, int d)
    {
        float val = 0;

        for (int i = 0; i < d; ++i)
            val += p[i]*q[i];

        return 1.0 - val;
    }
};

struct ChebyshevDistance
{
    float operator()(const float *p, const float *q, int d)
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
