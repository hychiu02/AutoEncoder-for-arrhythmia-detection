#include <sstream>
#include <fstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "neuron.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_neuron, mod)
{
    mod.doc() = "Neuron binding";

    py::class_<Neuron>(mod, "Neuron")
        .def(py::init<>())
        .def(py::init<Neuron const &>())
        .def(py::init<std::size_t, bool, double>(), py::arg("num_inputs"), py::arg("init_with_constant_weights") = true, py::arg("constant_weight") = 0.5)
        .def("__str__", [](Neuron & n)
        {
            std::stringstream ss;
            ss << n;

            return ss.str();
        })
        .def("forward", &Neuron::forward)
        .def("get_output", &Neuron::get_output)
        .def("update_grad", &Neuron::update_grad)
        .def("update_weights", &Neuron::update_weights)
        // Save and load need a file pointer, but file pointer will pass from MLP level, so wrap these functions for testing
        .def(
            "test_save_neuron",
            [](Neuron & n, std::string fname)
            {
                FILE * file;
                file = fopen(fname.c_str(), "wb");
                n.save_neuron(file);
                fclose(file);
            }
        )
        .def(
            "test_load_neuron",
            [](Neuron & n, std::string fname)
            {
                FILE * file;
                file = fopen(fname.c_str(), "rb");
                n.load_neuron(file);
                fclose(file);
            }
        );

}