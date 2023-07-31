#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <KDTree/KDTree.h>
#include <KDTree/Queries/LineNNQuery.h>
#include <KDTree/Queries/LineRadiusQuery.h>
#include <KDTree/Queries/LinekNNQuery.h>
#include <KDTree/Queries/PointNNQuery.h>
#include <KDTree/Queries/PointRadiusQuery.h>
#include <KDTree/Queries/PointkNNQuery.h>
#include <KDTree/Queries/AlignedRectQuery.h>

using namespace kdtree;
namespace py = pybind11;
using namespace py::literals;

namespace {

template <int DIM>
void
add_kd_tree(py::module &m, const char *name, int dim) {

    using Tree = KDTree<DIM>;

    auto pykdtree2d =
        py::class_<Tree, std::shared_ptr<Tree>>(m, name)
            .def(pybind11::init<>(
                [dim](py::array_t<float,
                                  py::array::c_style | py::array::forcecast>
                          points) {
                    py::buffer_info buf = points.request();

                    Tree tree(dim);

                    if (buf.ndim != 2 || buf.shape[1] != tree.get_dim()) {
                        throw std::runtime_error("Shape must be (x, dim)");
                    }

                    float *ptr = static_cast<float *>(buf.ptr);

                    tree.build(ptr, ptr + buf.shape[0] * buf.shape[1]);
                    return tree;
                }))
            .def("get_size", &Tree::get_size)
            .def("get_dim", &Tree::get_dim)
            .def("get_data",
                 [](Tree &tree) {
                     py::array_t<float> arr(
                         {tree.get_size(), (size_t) tree.get_dim()});
                     auto rw = arr.mutable_unchecked<2>();
                     for (int i = 0; i < tree.get_size(); ++i) {
                         for (int j = 0; j < tree.get_dim(); ++j) {
                             rw(i, j) = tree.get_data()[i * tree.get_dim() + j];
                         }
                     }
                     return arr;
                 })
            .def("get_indices",
                 [](Tree &tree) {
                     py::array_t<int> arr(tree.get_size());
                     auto rw = arr.mutable_unchecked<1>();
                     for (int i = 0; i < tree.get_size(); ++i) {
                         rw(i) = tree.get_indices()[i];
                     }
                     return arr;
                 })
            .def("point_nn_search",
                 [](Tree &tree, const Eigen::VectorXf &point) {
                     if (point.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     PointNNQuery<Tree> q(tree);
                     auto res = q.search(point.data()).getResult();
                     return py::make_tuple(tree.get_index(res.pointer),
                                           res.distance);
                 })
            .def("point_knn_search",
                 [](Tree &tree, const Eigen::VectorXf &point, int k) {
                     if (point.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     PointkNNQuery<Tree> q(tree, k);
                     auto res = q.search(point.data()).getResult();
                     py::list rl;
                     for (auto &r : res.results) {
                         rl.append(py::make_tuple(tree.get_index(r.pointer),
                                                  r.distance));
                     }
                     return rl;
                 })
            .def("point_radius_search",
                 [](Tree &tree, const Eigen::VectorXf &point, float radius) {
                     if (point.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     PointRadiusQuery<Tree> q(tree, radius);
                     auto res = q.search(point.data()).getResult();
                     py::list rl;
                     for (auto &r : res.results) {
                         rl.append(py::make_tuple(tree.get_index(r.pointer),
                                                  r.distance));
                     }
                     return rl;
                 })
            .def("aligned_rect_search",
                 [](Tree &tree, const Eigen::VectorXf &lower, const Eigen::VectorXf &upper) {
                     if (lower.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     if (upper.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     AlignedRectQuery<Tree> q(tree);
                     auto res = q.search(lower.data(), upper.data()).getResult();
                     py::list rl;
                     for (auto &r : res.results) {
                         rl.append(tree.get_index(r.pointer));
                     }
                     return rl;
                 })
            .def("line_nn_search",
                 [](Tree &tree, const Eigen::VectorXf &from,
                    const Eigen::VectorXf &to) {
                     if (from.size() != tree.get_dim() ||
                         to.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     LineNNQuery<Tree> q(tree);
                     auto res = q.search(from.data(), to.data()).getResult();
                     return py::make_tuple(tree.get_index(res.pointer),
                                           res.distance);
                 })
            .def("line_knn_search",
                 [](Tree &tree, const Eigen::VectorXf &from,
                    const Eigen::VectorXf &to, int k) {
                     if (from.size() != tree.get_dim() ||
                         to.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     LinekNNQuery<Tree> q(tree, k);
                     auto res = q.search(from.data(), to.data()).getResult();
                     py::list rl;
                     for (auto &r : res.results) {
                         rl.append(py::make_tuple(tree.get_index(r.pointer),
                                                  r.distance));
                     }
                     return rl;
                 })
            .def("line_radius_search",
                 [](Tree &tree, const Eigen::VectorXf &from,
                    const Eigen::VectorXf &to, float radius) {
                     if (from.size() != tree.get_dim() ||
                         to.size() != tree.get_dim()) {
                         throw std::runtime_error(
                             "Point dimension must match tree dimension");
                     }
                     LineRadiusQuery<Tree> q(tree, radius);
                     auto res = q.search(from.data(), to.data()).getResult();
                     py::list rl;
                     for (auto &r : res.results) {
                         rl.append(py::make_tuple(tree.get_index(r.pointer),
                                                  r.distance));
                     }
                     return rl;
                 });

    py::class_<LineNNQuery<Tree>>(pykdtree2d, "LineNNQuery");
    py::class_<PointkNNQuery<Tree>>(pykdtree2d, "PointkNNQuery");
    py::class_<PointNNQuery<Tree>>(pykdtree2d, "PointNNQuery");
    py::class_<PointRadiusQuery<Tree>>(pykdtree2d, "PointRadiusQuery");
    py::class_<AlignedRectQuery<Tree>>(pykdtree2d, "AlignedRectQuery");
}

};   // namespace

PYBIND11_MODULE(PyKDTree, m) {
    m.doc() = "pykdtree test module";
    add_kd_tree<-1>(m, "KDTree", -1);
    add_kd_tree<2>(m, "KDTree2d", 2);
    add_kd_tree<3>(m, "KDTree3d", 3);
    add_kd_tree<4>(m, "KDTree4d", 4);
}