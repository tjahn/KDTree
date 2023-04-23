#include <iostream>
#include <array>

#include <KDTree/KDTree.h>
#include <KDTree/Queries/PointNNQuery.h>

using namespace std;
using namespace kdtree;

vector<float> random_points(int N, int DIM)
{
    vector<float> points(N * DIM);
    for (auto &v : points)
        v = 1. * rand() / RAND_MAX;
    return points;
}

int main()
{
    std::cout << "Start" << std::endl;
    srand(0);

    const bool multithreaded = true;
    const int N = 100000;

    using Db = kdtree::KDTree<2>;
    Db tree;

    const auto data = random_points(N, tree.get_dim());
    tree.build(data.begin(), data.end(), multithreaded);
    std::cout << "Build db" << std::endl;

    if (true)
    {
        // query one point
        kdtree::PointNNQuery<Db> query(tree);
        std::array<float, 2> target{0.1f, 0.15f};
        auto res = query.search(target.data());
        std::cout << "(" << res.pointer[0] << ", " << res.pointer[1] << ") " << res.distance << std::endl;
    }

    std::cout << "End" << std::endl;
    return 0;
}
