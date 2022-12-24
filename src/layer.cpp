#include <sstream>
#include <fstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "layer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_layer, mod)
{
    mod.doc() = "Layer binding";

    py::class_<Layer>(mod, "Layer")
        .def(py::init<>())
        .def(py::init<std::size_t, std::size_t, std::string, bool, double>(), py::arg("num_inputs"), py::arg("num_neurons"), py::arg("activation_func_name"), py::arg("init_with_constant_weights") = true, py::arg("constant_weight") = 0.5)
        .def("__str__", [](Layer & n)
        {
            std::stringstream ss;
            ss << n;

            return ss.str();
        })
        .def(
            "test_save_layer",
            [](Layer & n, std::string fname)
            {
                FILE * file;
                file = fopen(fname.c_str(), "wb");
                n.save_layer(file);
                fclose(file);
            }
        )
        .def(
            "test_load_layer",
            [](Layer & n, std::string fname)
            {
                FILE * file;
                file = fopen(fname.c_str(), "rb");
                n.load_layer(file);
                fclose(file);
            }
        );

}