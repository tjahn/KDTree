#include <array>
#include <iostream>

#include <KDTree/KDTree.h>
#include <KDTree/Printer.h>
#include <KDTree/Queries/AlignedRectQuery.h>
#include <KDTree/Queries/PointNNQuery.h>
#include <KDTree/Queries/PointRadiusQuery.h>
#include <KDTree/Queries/PointkNNQuery.h>

using namespace std;
using namespace kdtree;

vector<float>
random_points(int N, int DIM) {
    vector<float> points(N * DIM);
    for (auto &v : points)
        v = 1. * rand() / RAND_MAX;
    return points;
}

int
main() {
    std::cout << "Start" << std::endl;
    srand(0);

    const bool multithreaded = false;
    const int N = 1e4;

    using Db = kdtree::KDTree<2>;
    Db tree;

    const auto data = random_points(N, tree.get_dim());
    tree.build(data.begin(), data.end(), multithreaded);
    std::cout << "Build db" << std::endl;

    std::array<float, 2> target{0.21f, 0.15f};

    if (N <= 20) {
        std::cout << "TARGET (" << target[0];
        for (int i = 1; i < target.size(); ++i)
            std::cout << ", " << target[i];
        std::cout << ")" << std::endl;
        kdtree::KDTreePrinter(tree).print();
    }

    double dist;

    if (true) {
        std::cout << "nn query" << std::endl;
        kdtree::PointNNQuery<Db> query(tree);
        auto &res = query.search(target.data()).getResult();
        std::cout << "(" << res.pointer[0] << ", " << res.pointer[1] << ") "
                  << res.distance << std::endl;
        dist = std::sqrt(res.distance);
    }

    if (true) {
        std::cout << "knn query" << std::endl;
        kdtree::PointkNNQuery<Db> query(tree, 3);
        auto &res = query.search(target.data()).getResult();
        for (int i = 0; i < res.results.size(); ++i) {
            auto &r = res.results[i];
            std::cout << i << " (" << r.pointer[0] << ", " << r.pointer[1]
                      << ") " << r.distance << std::endl;
        }
        dist = std::sqrt(res.results.back().distance);
    }

    if (true) {
        std::cout << "radius query " << dist << std::endl;
        kdtree::PointRadiusQuery<Db> query(tree, dist, true);
        auto &res = query.search(target.data()).getResult();
        for (int i = 0; i < res.results.size(); ++i) {
            auto &r = res.results[i];
            std::cout << i << " (" << r.pointer[0] << ", " << r.pointer[1]
                      << ") " << r.distance << std::endl;
        }
    }

    if (true)
    {
        std::cout << "aligned rect query " << dist << std::endl;
        std::array<float, 2> target2 = target;
        for (auto &v : target2)
            v += std::sqrt(5. / N);

        kdtree::AlignedRectQuery<Db> query(tree);
        auto &res = query.search(target.data(), target2.data()).getResult();
        for (int i = 0; i < res.results.size(); ++i)
        {
            auto &r = res.results[i];
            std::cout << i << " (" << r.pointer[0] << ", " << r.pointer[1] << ") " << std::endl;
        }
    }

    std::cout << "End" << std::endl;
    return 0;
}
