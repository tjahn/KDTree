CPMAddPackage("gh:pybind/pybind11@2.11.1")


find_package (Eigen3 REQUIRED NO_MODULE)

pybind11_add_module(PyKDTree THIN_LTO PyKDTree.cpp)

target_link_libraries(PyKDTree PUBLIC 
kdtree
Eigen3::Eigen
)
