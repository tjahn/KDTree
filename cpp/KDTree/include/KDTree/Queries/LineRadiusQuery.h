#pragma once

#include <limits>

namespace kdtree {

template <typename Db> class LineRadiusQuery {
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
    LineRadiusQuery(const Db &db, const Scalar radius, bool ordered = false)
        : db(db), radius(radius), radiusSqr(radius * radius), ordered(ordered),
          dim(db.get_dim()) {}

    LineRadiusQuery &search(const Scalar *from, const Scalar *to) {
        res.results.clear();

        Scalar segmentLengthSquared = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            Scalar diff = to[i] - from[i];
            segmentLengthSquared += diff * diff;
        }

        perform_search(db.get_navigator(), 0, from, to,
                       std::sqrt(segmentLengthSquared));

        if (ordered) {
            std::sort(res.results.begin(), res.results.end(),
                      [](auto &a, auto &b) { return a.distance < b.distance; });
        }
        return *this;
    }

    const Result &getResult() const { return res; }

    const Db &getDb() const { return db; }

  private:
    const Db &db;
    const int dim;
    const Scalar radius;
    const Scalar radiusSqr;
    const bool ordered;

    Result res;

  private:
    inline void perform_bruteforce_search(typename Db::Navigator nav,
                                          int blocksize, const Scalar *from,
                                          const Scalar *to,
                                          const Scalar segmentLengthSquared) {
        for (int pid = 0; pid < blocksize; ++pid) {
            auto ptr = nav.pointer + pid * dim;
            // calculate distance of current point
            Scalar dist2 =
                distanceSqrPointToLine(from, to, ptr, segmentLengthSquared);

            // if current point is close enough
            if (dist2 <= radiusSqr) {
                res.results.emplace_back(dist2, ptr);
            }
        }
    }

    void perform_search(typename Db::Navigator nav, int depth,
                        const Scalar *_from, const Scalar *_to,
                        double segmentLength) {
        const int blocksize = nav.blocksize;

        if (blocksize == 0) {
            return;
        }

        if (blocksize <= Db::BLOCK_SIZE) {
            perform_bruteforce_search(nav, blocksize, _from, _to,
                                      segmentLength * segmentLength);
            return;
        }

        const auto ptr = nav.pointer;

        const auto axis = depth % dim;
        const auto fromIsSmaller = _from[axis] <= _to[axis];
        const auto lower = fromIsSmaller ? _from : _to;
        const auto upper = fromIsSmaller ? _to : _from;

        const Scalar dxl = ptr[axis] - lower[axis];
        const Scalar dxt = ptr[axis] - upper[axis];

        // calculate distance of current point
        const Scalar dist2 = distanceSqrPointToLine(
            lower, upper, ptr, segmentLength * segmentLength);
        // if current point is close enough take it
        if (dist2 <= radiusSqr) {
            res.results.emplace_back(dist2, nav.pointer);
        }

        if (dxl < -radius) {
            // the whole query is to the right of the current point

            // 1 on the left make a point distance search to the lower point
            //   - skip if current best radius is smaller than dxl**2
            if (dxl * dxl <= radiusSqr) {
                perform_search(nav.get_left(dim), depth + 1, lower, upper,
                               segmentLength);
            }

            perform_search(nav.get_right(dim), depth + 1, lower, upper,
                           segmentLength);
            return;
        } else if (dxt > radius) {
            // the whole query is to the left of the current point

            // 1 on the left make a line-distance search
            perform_search(nav.get_left(dim), depth + 1, lower, upper,
                           segmentLength);

            // 2 on the right make a point distance search  to the lower point
            //   - skip if current best radius is smaller than dxt**2
            if (dxt * dxt <= radiusSqr) {
                perform_search(nav.get_right(dim), depth + 1, lower, upper,
                               segmentLength);
            }
            return;
        } else {

            // the query is to the left and right of the current point
            // since we have rounded caps we need some safety margin for cutting

            if (upper[axis] == lower[axis]) {
                // handle upper[axis]-lower[axis] == 0
                perform_search(nav.get_left(dim), depth + 1, lower, upper,
                               segmentLength);
                perform_search(nav.get_right(dim), depth + 1, lower, upper,
                               segmentLength);
                return;
            }

            // 1 intersect lower-upper at the current axis at
            // nav.pointer[axis]
            Scalar intersection[dim];
            const Scalar iumlx = 1 / (upper[axis] - lower[axis]);

            // 2 on the left make line query with lower-intersection
            // TODO dont go back whole radius, but use angle
            const Scalar dlx = std::min(Scalar(1), (dxl + radius) * iumlx);
            for (int i = 0; i < dim; ++i) {
                intersection[i] = lower[i] + dlx * (upper[i] - lower[i]);
            }
            perform_search(nav.get_left(dim), depth + 1, lower, intersection,
                           segmentLength * dlx);

            // 3 on the right make line query with intersection-upper
            // TODO dont go back whole radius, but use angle
            const Scalar dtx = std::min(Scalar(1), (-dxt + radius) * iumlx);
            for (int i = 0; i < dim; ++i) {
                intersection[i] = upper[i] + dtx * (lower[i] - upper[i]);
            }
            perform_search(nav.get_right(dim), depth + 1, intersection, upper,
                           segmentLength * dtx);
            return;
        }
    }

  private:
    inline void perform_bruteforce_search_point_radius(
        typename Db::Navigator nav, const int blocksize, const Scalar *query) {
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
            if (dist2 < radiusSqr) {
                res.results.emplace_back(dist2, ptr);
            }
        }
    }

    Scalar distanceSqrPointToLine(const Scalar *from, const Scalar *to,
                                  const Scalar *point,
                                  Scalar segmentLengthSquared) const {
        Scalar distance = 0.0;

        if (segmentLengthSquared == 0.0) {
            // The line segment is actually a point
            for (size_t i = 0; i < dim; ++i) {
                Scalar diff = point[i] - from[i];
                distance += diff * diff;
            }
        } else {
            Scalar t = 0.0;
            const Scalar isql = 1 / segmentLengthSquared;
            for (size_t i = 0; i < dim; ++i) {
                t += ((point[i] - from[i]) * (to[i] - from[i])) * isql;
            }

            if (t < 0.0) {
                // The closest point is outside the line segment, clos to 'from'
                for (size_t i = 0; i < dim; ++i) {
                    Scalar diff = point[i] - from[i];
                    distance += diff * diff;
                }
            } else if (t > 1.0) {
                // The closest point is outside the line segment, close to 'to'
                for (size_t i = 0; i < dim; ++i) {
                    Scalar diff = point[i] - to[i];
                    distance += diff * diff;
                }
            } else {
                // The closest point is within the line segment
                for (size_t i = 0; i < dim; ++i) {
                    Scalar projection = from[i] + t * (to[i] - from[i]);
                    Scalar diff = point[i] - projection;
                    distance += diff * diff;
                }
            }
        }

        return distance;
    }
};
}   // namespace kdtree
