#include <benchmark/benchmark.h>

#include <KDTree/KDTree.h>
#include <KDTree/Queries/LineNNQuery.h>
#include <KDTree/Queries/LineRadiusQuery.h>
#include <KDTree/Queries/LinekNNQuery.h>

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
    kdtree::LineNNQuery<Db2d> lineNNQuery;
    kdtree::LinekNNQuery<Db2d> line2NNQuery;
    kdtree::LinekNNQuery<Db2d> line8NNQuery;
    kdtree::LineRadiusQuery<Db2d> lineRadiusQuery;

    int numQueries = 1000;
    vector<float> queries;

    MyDbFixture()
        : lineNNQuery(tree), line2NNQuery(tree, 2), line8NNQuery(tree, 8),
          lineRadiusQuery(tree, 0.01) {}

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
BENCHMARK_DEFINE_F(MyDbFixture, LineNNQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % (numQueries - 1);
        auto &res = lineNNQuery
                        .search(queries.data() + tree.get_dim() * i,
                                queries.data() + tree.get_dim() * (i + 1))
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, LineNNQuery)
    ->Arg(1e3)
    ->Arg(1e4)
    ->Arg(1e5)
    ->Arg(1e6);

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, Line2NNQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % (numQueries - 1);
        auto &res = line2NNQuery
                        .search(queries.data() + tree.get_dim() * i,
                                queries.data() + tree.get_dim() * (i + 1))
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, Line2NNQuery)->Arg(1e3)->Arg(1e4)->Arg(1e5);

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, Line8NNQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % (numQueries - 1);
        auto &res = line8NNQuery
                        .search(queries.data() + tree.get_dim() * i,
                                queries.data() + tree.get_dim() * (i + 1))
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, Line8NNQuery)->Arg(1e3)->Arg(1e4)->Arg(1e5);

// test queries
BENCHMARK_DEFINE_F(MyDbFixture, LineRadiusQuery)
(benchmark::State &st) {
    int i = 0;
    for (auto _ : st) {
        i = (i++) % (numQueries - 1);
        auto &res = lineRadiusQuery
                        .search(queries.data() + tree.get_dim() * i,
                                queries.data() + tree.get_dim() * (i + 1))
                        .getResult();
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK_REGISTER_F(MyDbFixture, LineRadiusQuery)
    ->Arg(1e3)
    ->Arg(1e4)
    ->Arg(1e5)
    ->Arg(1e6);

BENCHMARK_MAIN();
