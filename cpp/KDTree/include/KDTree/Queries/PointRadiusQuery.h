#pragma once

#include <limits>

namespace kdtree
{

    template <typename Db>
    class PointRadiusQuery
    {
    public:
        using Scalar = typename Db::Scalar;

        struct Result
        {
            struct SingleResult
            {
                Scalar distance;
                const Scalar *pointer;
                SingleResult() {}

                SingleResult(Scalar distance,
                             const Scalar *pointer) : distance(distance), pointer(pointer) {}
            };

            std::vector<SingleResult> results;
        };

    public:
        PointRadiusQuery(const Db &db, double radius,
                         bool ordered = false) : db(db), radius(radius), radiusSqr(radius * radius), ordered(ordered) {}

        PointRadiusQuery &search(const Scalar *_query)
        {
            query = _query;
            res.results.clear();
            perform_search(db.get_navigator(), 0);
            if (ordered)
            {
                std::sort(res.results.begin(), res.results.end(), [](auto &a, auto &b)
                          { return a.distance < b.distance; });
            }
            return *this;
        }

        const Result &getResult() const
        {
            return res;
        }

    private:
        const Db &db;
        const Scalar *query;
        const double radius;
        const double radiusSqr;
        bool ordered;

        Result res;

    private:
        inline void perform_bruteforce_search(typename Db::Navigator nav, const int blocksize)
        {
            const int dim = db.get_dim();

            for (int pid = 0; pid < blocksize; ++pid)
            {
                auto ptr = nav.pointer + pid * dim;

                // calculate distance of current point
                Scalar dist2 = 0;
                for (int i = 0; i < dim; ++i)
                {
                    auto val = query[i] - ptr[i];
                    dist2 += val * val;
                }

                // if current point is new best take it
                if (dist2 < radiusSqr)
                {
                    res.results.emplace_back(dist2, ptr);
                }
            }
        }

        void perform_search(typename Db::Navigator nav, int depth)
        {
            const int blocksize = nav.blocksize;

            if (blocksize == 0)
                return;

            if (blocksize <= Db::BLOCK_SIZE)
            {
                perform_bruteforce_search(nav, blocksize);
                return;
            }

            const int dim = db.get_dim();

            // calculate distance of current point
            Scalar dist2 = 0;
            for (int i = 0; i < dim; ++i)
            {
                auto val = query[i] - nav.pointer[i];
                dist2 += val * val;
            }

            // if current point is new best take it
            if (dist2 < radiusSqr)
            {
                res.results.emplace_back(dist2, nav.pointer);
            }

            // check left/right if necessary
            auto axis = depth % dim;
            Scalar dx = query[axis] - nav.pointer[axis];

            if (dx < radius)
                perform_search(nav.get_left(dim), depth + 1);
            if (dx > -radius)
                perform_search(nav.get_right(dim), depth + 1);
        }
    };
}
