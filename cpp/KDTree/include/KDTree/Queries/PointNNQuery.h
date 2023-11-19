#pragma once

#include <limits>

namespace kdtree {

template <typename Db> class PointNNQuery {
  public:
    using Scalar = typename Db::Scalar;

    struct Result {
        Scalar distance;
        const Scalar *pointer;
    };

  public:
    PointNNQuery(const Db &db) : db(db) {}

    PointNNQuery &search(const Scalar *_query) {
        query = _query;
        res.distance = std::numeric_limits<Scalar>::max();
        res.pointer = nullptr;
        perform_search(db.get_navigator(), 0);
        return *this;
    }

    const Result &getResult() const { return res; }

  private:
    const Db &db;
    const Scalar *query;

    Result res;

  private:
    inline void perform_bruteforce_search(typename Db::Navigator nav,
                                          int blocksize) {
        const int dim = db.get_dim();

        for (int pid = 0; pid < blocksize; ++pid) {
            auto ptr = nav.pointer + pid * dim;

            // calculate distance of current point
            Scalar dist2 = 0;
            for (int i = 0; i < dim; ++i) {
                auto val = query[i] - ptr[i];
                dist2 += val * val;
            }

            // if current point is new best take it
            if (dist2 < res.distance) {
                res.distance = dist2;
                res.pointer = ptr;
            }
        }
    }

    void perform_search(typename Db::Navigator nav, int depth) {
        const int blocksize = nav.blocksize;

        if (blocksize == 0)
            return;

        if (blocksize <= Db::BLOCK_SIZE) {
            perform_bruteforce_search(nav, blocksize);
            return;
        }

        const int dim = db.get_dim();

        // calculate distance of current point
        Scalar dist2 = 0;
        for (int i = 0; i < dim; ++i) {
            auto val = query[i] - nav.pointer[i];
            dist2 += val * val;
        }

        // if current point is new best take it
        if (dist2 < res.distance) {
            res.distance = dist2;
            res.pointer = nav.pointer;
        }

        // check left/right if necessary
        auto axis = depth % dim;
        Scalar dx = query[axis] - nav.pointer[axis];

        if (dx <
            0) {   // nearest neighbor most likely is left. look there first
            perform_search(nav.get_left(dim), depth + 1);
            if (dx * dx < res.distance) {
                perform_search(nav.get_right(dim), depth + 1);
            }
        } else {   // nearest neighbor most likely is right. look there first
            perform_search(nav.get_right(dim), depth + 1);
            if ((dx * dx < res.distance)) {
                perform_search(nav.get_left(dim), depth + 1);
            }
        }
    }
};
}   // namespace kdtree
