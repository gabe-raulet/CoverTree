template <class Index, class Real, class Atom>
class Euclidean : public MetricSpace<Index, Real, Atom>
{
    public:

        virtual constexpr const char* metric() const final { return "euclidean"; }

        using MetricSpace<Index, Real, Atom>::MetricSpace;
        using MetricSpace<Index, Real, Atom>::distance;

        virtual Real distance(const Atom *p, const Atom *q) const final
        {
            Real val = 0;
            Index d = this->num_dimensions();

            for (Index i = 0; i < d; ++i)
            {
                Real delta = p[i] - q[i];
                val += delta * delta;
            }

            return std::sqrt(val);
        }
};

template <class Index, class Real, class Atom>
class Manhattan : public MetricSpace<Index, Real, Atom>
{
    public:

        virtual constexpr const char* metric() const final { return "manhattan"; }

        using MetricSpace<Index, Real, Atom>::MetricSpace;
        using MetricSpace<Index, Real, Atom>::distance;

        virtual Real distance(const Atom *p, const Atom *q) const final
        {
            Real val = 0;
            Index d = this->num_dimensions();

            for (Index i = 0; i < d; ++i)
            {
                val += std::abs(p[i] - q[i]);
            }

            return val;
        }
};

template <class Index, class Real, class Atom>
class Chebyshev : public MetricSpace<Index, Real, Atom>
{
    public:

        virtual constexpr const char* metric() const final { return "chebyshev"; }

        using MetricSpace<Index, Real, Atom>::MetricSpace;
        using MetricSpace<Index, Real, Atom>::distance;

        virtual Real distance(const Atom *p, const Atom *q) const final
        {
            Real val = 0;
            Index d = this->num_dimensions();

            for (Index i = 0; i < d; ++i)
            {
                val = std::max(val, static_cast<Real>(std::abs(p[i] - q[i])));
            }

            return val;
        }
};

template <class Index, class Real, class Atom>
class Levenshtein : public MetricSpace<Index, Real, Atom>
{
    public:

        virtual constexpr const char* metric() const final { return "levenshtein"; }

        using MetricSpace<Index, Real, Atom>::MetricSpace;
        using MetricSpace<Index, Real, Atom>::distance;

        virtual Real distance(const Atom *p, const Atom *q) const final
        {
            Index d = this->num_dimensions();
            typename MetricSpace<Index, Real, Atom>::IndexVector v0(d+1), v1(d+1);

            for (Index i = 0; i <= d; ++i)
                v0[i] = i;

            for (Index i = 0; i < d; ++i)
            {
                v1[0] = i+1;

                for (Index j = 0; j < d; ++j)
                {
                    Index del_cost = v0[j+1] + 1;
                    Index ins_cost = v1[j] + 1;
                    Index sub_cost = (p[i] == q[i])? v0[j] : v0[j] + 1;

                    v1[j+1] = std::min(del_cost, std::min(ins_cost, sub_cost));
                }

                std::swap(v0, v1);
            }

            return v0[d];
        }
};

