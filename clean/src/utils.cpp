#include "utils.h"

void selection_sample(Index range, Index size, IndexVector& sample, int seed)
{
    /*
     * Reference for random sampling: https://bastian.rieck.me/blog/2017/selection_sampling/
     */

    static std::random_device rd;
    seed = seed < 0 ? rd() : seed;
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> U(0, std::nextafter(1.0, std::numeric_limits<double>::max()));

    sample.clear();
    sample.reserve(size);

    for (Index i = 0; i < range; ++i)
    {
        if ((range - i) * U(gen) < size - sample.size())
            sample.push_back(i);

        if (sample.size() == size)
            break;
    }
}
