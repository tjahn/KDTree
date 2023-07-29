#include <iostream>
#include <array>

#include <benchmark/benchmark.h>

#include <KDTree/KDTree.h>
#include <KDTree/Queries/PointRadiusQuery.h>
#include <KDTree/Queries/PointkNNQuery.h>
#include <KDTree/Queries/PointNNQuery.h>

using namespace std;
using namespace kdtree;

using Db2d = kdtree::KDTree<2>;

vector<float> random_points(int N, int DIM)
{
    vector<float> points(N * DIM);
    for (auto &v : points)
        v = 1. * rand() / RAND_MAX;
    return points;
}

class MyRandomPointsFixture : public benchmark::Fixture
{
public:
    Db2d tree;

    int N;
    std::vector<float> data;

    void SetUp(const ::benchmark::State &state)
    {
        N = state.range(0);
        data = random_points(N, tree.get_dim());
    }

    void TearDown(const ::benchmark::State &state)
    {
    }
};

// test build tree
BENCHMARK_DEFINE_F(MyRandomPointsFixture, BuildTree)
(benchmark::State &st)
{
    const bool multithreaded = st.range(1);
    for (auto _ : st)
    {
        tree.build(data.begin(), data.end(), multithreaded);
    }
}
BENCHMARK_REGISTER_F(MyRandomPointsFixture, BuildTree)
    ->Args({long(1e4), 0})
    ->Args({long(1e4), 1})

    ->Args({long(1e5), 0})
    ->Args({long(1e5), 1})

    ->Args({long(1e6), 0})
    ->Args({long(1e6), 1});

BENCHMARK_MAIN();
