#pragma once

#include <iostream>

namespace kdtree {

template <typename Db> class KDTreePrinter {
  public:
    using Scalar = typename Db::Scalar;

    KDTreePrinter(const Db &db) : db(db) {}

    void print() { perform_print(db.get_navigator(), 0); }

  private:
    const Db &db;

  private:
    void perform_print(typename Db::Navigator nav, int depth) {
        if (nav.blocksize == 0) {
            return;
        }

        const int dim = db.get_dim();
        const int axis = depth % dim;

        std::cout << depth << " ";
        for (int i = 0; i < depth; ++i)
            std::cout << " ";
        std::cout << "(" << nav.pointer[0];
        for (int i = 1; i < db.get_dim(); ++i)
            std::cout << ", " << nav.pointer[i];
        std::cout << ")\n";

        perform_print(nav.get_left(dim), depth + 1);
        perform_print(nav.get_right(dim), depth + 1);
    }
};

}   // namespace kdtree