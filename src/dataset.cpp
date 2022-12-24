#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "mlp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_dataset, mod)
{
    mod.doc() = "Dataset binding";

    py::class_<Dataset>(mod, "Dataset")
        .def(py::init<std::vector<std::string>>())
        .def("__str__", [](Dataset & self)
        {
            std::stringstream ss;
            ss << self;

            return ss.str();
        })
        .def("__len__", [](Dataset & self)
        {
            return self.get_dataset_size();
        })
        .def("get_input_vector", &Dataset::get_input_vector)
        .def("get_input_size", &Dataset::get_input_size)
        .def("get_output_vector", &Dataset::get_output_vector)
        .def("get_output_size", &Dataset::get_output_size);

    py::class_<Gate_Dataset, Dataset>(mod, "Gate_Dataset")
        .def(py::init<std::vector<std::string>>())
        .def("__str__", [](Gate_Dataset & self)
        {
            std::stringstream ss;
            ss << self;

            return ss.str();
        })
        .def("__len__", [](Gate_Dataset & self)
        {
            return self.get_gata_dataset_size();
        })
        .def("load_data", &Gate_Dataset::load_data)
        .def("get_input_vector", &Gate_Dataset::get_input_vector)
        .def("get_input_size", &Gate_Dataset::get_input_size)
        .def("get_output_vector", &Gate_Dataset::get_output_vector)
        .def("get_output_size", &Gate_Dataset::get_output_size);

    py::class_<VA_Dataset, Dataset>(mod, "VA_Dataset")
        .def(py::init<std::vector<std::string>>())
        .def("__str__", [](VA_Dataset & self)
        {
            std::stringstream ss;
            ss << self;

            return ss.str();
        })
        .def("__len__", [](Gate_Dataset & self)
        {
            return self.get_gata_dataset_size();
        })
        .def("load_data", &VA_Dataset::load_data)
        .def("get_input_vector", &VA_Dataset::get_input_vector)
        .def("get_input_size", &VA_Dataset::get_input_size)
        .def("get_output_vector", &VA_Dataset::get_output_vector)
        .def("get_output_size", &VA_Dataset::get_output_size);
}