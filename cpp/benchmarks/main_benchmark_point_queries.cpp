
#include <benchmark/benchmark.h>

#include <KDTree/KDTree.h>
#include <KDTree/Queries/PointNNQuery.h>
#include <KDTree/Queries/PointRadiusQuery.h>
#include <KDTree/Queries/PointkNNQuery.h>

using namespace std;
using namespace kdtree;

using Db2d = kdtree::KDTree<2>;
using Db = kdtree::KDTree<-1>;

vector<float>
random_points(int N, int DIM) {
    vector<float> points(N * DIM);
    for (auto &v : points)
        v = 1. * rand() / RAND_MAX;
    return points;
}

class MyDbFixture : public benchmark::Fixture {
  public:
    Db2d tree;
    kdtree::PointNNQuery<Db2d> pointNNQuery;
    kdtree::PointkNNQuery<Db2d> point2NNQuery;
    kdtree::PointkNNQuery<Db2d> point8NNQuery;
    kdtree::PointRadiusQuery<Db2d> pointRadiusQuery;

    int numQueries = 1000;
    vector<float> queries;

    MyDbFixture()
        : pointNNQuery(tree), point2NNQuery(tree, 2), point8NNQuery(tree, 8),
          pointRadiusQuery(tree, 0.01) {}

    void SetUp(const ::benchmark::State &state) {
        int N = state.range(0);
        bool multithreaded = true;
        const auto data = random_points(N, tree.get_dim());
        tree.build(data.begin(), data.end(), multithreaded);
        queries = random_points(numQueries, tree.get_dim());
    }

    void TearDown(const ::benchmark::State &state) {}
};

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, PointNNQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % numQueries;
        auto &res = pointNNQuery.search(queries.data() + tree.get_dim() * i)
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, PointNNQuery)
    ->Arg(1e3)
    ->Arg(1e4)
    ->Arg(1e5)
    ->Arg(1e6);

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, Point2NNQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % numQueries;
        auto &res = point2NNQuery.search(queries.data() + tree.get_dim() * i)
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, Point2NNQuery)
    ->Arg(1e3)
    ->Arg(1e4)
    ->Arg(1e5)
    ->Arg(1e6);

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, Point8NNQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % numQueries;
        auto &res = point8NNQuery.search(queries.data() + tree.get_dim() * i)
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, Point8NNQuery)
    ->Arg(1e3)
    ->Arg(1e4)
    ->Arg(1e5)
    ->Arg(1e6);

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, PointRadiusQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % numQueries;
        auto &res = pointRadiusQuery.search(queries.data() + tree.get_dim() * i)
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, PointRadiusQuery)
    ->Arg(1e3)
    ->Arg(1e4)
    ->Arg(1e5)
    ->Arg(1e6);

BENCHMARK_MAIN();
