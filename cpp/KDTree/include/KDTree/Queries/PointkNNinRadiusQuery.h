#pragma once

#include <limits>

namespace kdtree {
template <typename Db> class PointkNNinRadiusQuery {
  public:
    using Scalar = typename Db::Scalar;

    struct Result {
        struct SingleResult {
            Scalar distance;
            const Scalar *pointer;
            SingleResult() {}

            SingleResult(Scalar distance, const Scalar *pointer)
                : distance(distance), pointer(pointer) {}
        };

        std::vector<SingleResult> results;
    };

  public:
    PointkNNinRadiusQuery(const Db &db, int k, Scalar radius) : db(db), k(k), radiusSqr(radius*radius) {}

    PointkNNinRadiusQuery &search(const Scalar *_query) {
        query = _query;
        res.results.clear();
        res.results.reserve(k);
        perform_search(db.get_navigator(), 0);
        return *this;
    }

    const Result &getResult() const { return res; }

  private:
    const Db &db;
    const Scalar *query;
    const int k;
    const Scalar radiusSqr;

    Result res;

  private:
    void perform_search(typename Db::Navigator nav, int depth) {
        int blocksize = nav.blocksize;

        if (blocksize == 0)
            return;

        const int dim = db.get_dim();

        // calculate distance of current point
        Scalar dist2 = 0;
        for (int i = 0; i < dim; ++i) {
            auto val = query[i] - nav.pointer[i];
            dist2 += val * val;
        }

        if (
            (dist2 <= radiusSqr)  // point is in radius
            &&
            (res.results.size() < k || dist2 < res.results.back().distance)// point is closer than current worst
            ) {

            if (res.results.size() >= 30) {
                // for large number of values it is better to keep the order by
                // inserting at the correct place
                res.results.emplace(
                    lower_bound(res.results.begin(), res.results.end(), dist2,
                                [](auto &a, auto b) { return a.distance < b; }),
                    dist2, nav.pointer);
            } else {
                // for small number of values its faster to just resort the
                // whole array
                res.results.emplace_back(dist2, nav.pointer);
                std::sort(
                    res.results.begin(), res.results.end(),
                    [](auto &a, auto &b) { return a.distance < b.distance; });
            }

            if (res.results.size() > k) {
                res.results.resize(k);
            }
        }

        // check left/right if necessary
        auto axis = depth % dim;
        Scalar dx = query[axis] - nav.pointer[axis];
        if (dx <
            0) {   // nearest neighbor most likely is left. look there first
            perform_search(nav.get_left(dim), depth + 1);
            if (dx * dx < res.results.back().distance)
                perform_search(nav.get_right(dim), depth + 1);
        } else {   // nearest neighbor most likely is right. look there first
            perform_search(nav.get_right(dim), depth + 1);
            if ((dx * dx < res.results.back().distance))
                perform_search(nav.get_left(dim), depth + 1);
        }
    }
};
}   // namespace kdtree
