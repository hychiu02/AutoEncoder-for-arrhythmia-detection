#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "mlp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_mlp, mod)
{
    mod.doc() = "MLP binding";

    py::class_<MLP>(mod, "MLP")
        .def(py::init<>())
        .def(py::init<const std::vector<std::size_t> &, const std::vector<std::string> &, bool, double>(), py::arg("num_layer_neurons"), py::arg("layer_activation_funcs"), py::arg("init_with_constant_weights") = false, py::arg("constant_weight") = 0.5)
        .def("__str__", [](MLP & n)
        {
            std::stringstream ss;
            ss << n;

            return ss.str();
        })
        .def("get_output", &MLP::get_output)
        .def("forward", &MLP::forward)
        .def("fit", &MLP::fit)
        .def("backward", &MLP::backward)
        .def("save_mlp", &MLP::save_mlp)
        .def("load_mlp", &MLP::load_mlp)
        .def("train", &MLP::train)
        .def("test", &MLP::test)
        .def("inference", &MLP::inference);
}