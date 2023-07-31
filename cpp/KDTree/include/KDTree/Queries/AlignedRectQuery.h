#pragma once

#include <limits>

namespace kdtree
{

    template <typename Db>
    class AlignedRectQuery
    {
    public:
        using Scalar = typename Db::Scalar;

        struct Result
        {
            struct SingleResult
            {
                const Scalar *pointer;
                SingleResult() {}

                SingleResult(
                    const Scalar *pointer) : pointer(pointer) {}
            };

            std::vector<SingleResult> results;
        };

    public:
        AlignedRectQuery(const Db &db) : db(db) {}

        AlignedRectQuery &search(const Scalar *_lower, const Scalar *_upper)
        {
            lower = _lower;
            upper = _upper;
            res.results.clear();
            perform_search(db.get_navigator(), 0);
            return *this;
        }

        const Result &getResult() const
        {
            return res;
        }

    private:
        const Db &db;
        const Scalar *lower;
        const Scalar *upper;

        Result res;

    private:
        inline void perform_bruteforce_search(typename Db::Navigator nav, const int blocksize)
        {
            const int dim = db.get_dim();

            for (int pid = 0; pid < blocksize; ++pid)
            {
                auto ptr = nav.pointer + pid * dim;

                // check if point is inside
                bool inside = true;
                for (int i = 0; i < dim; ++i)
                {
                    inside = !(!inside || ptr[i] < lower[i] || ptr[i] > upper[i]);
                }

                if (inside)
                {
                    res.results.emplace_back(ptr);
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
            const auto ptr = nav.pointer;

            // check if point is inside
            bool inside = true;
            for (int i = 0; i < dim; ++i)
            {
                inside = !(!inside || ptr[i] < lower[i] || ptr[i] > upper[i]);
            }

            if (inside)
            {
                res.results.emplace_back(ptr);
            }

            // check left/right if necessary
            const auto axis = depth % dim;

            if (lower[axis] < ptr[axis])
            {
                perform_search(nav.get_left(dim), depth + 1);
            }
            if (upper[axis] > ptr[axis])
            {
                perform_search(nav.get_right(dim), depth + 1);
            }
        }
    };
}
