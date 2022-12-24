#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_utils, mod)
{
    mod.doc() = "utils binding";

    py::class_<Activation>(mod, "Activation")
        .def(py::init<>())
        .def("get_activation_func", &Activation::get_activation_func)
        .def("get_deriv_activation_func", &Activation::get_deriv_activation_func);
    
    mod.def("mse", &mse);
    mod.def("deriv_mse", &deriv_mse);
}