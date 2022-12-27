CXX           := g++
CXXFLAGS      := -O3 -Wall -shared -std=c++17 -fPIC

HEADERPATH    := ./src
SOURCEPATH    := ./src

PYINCLUDE     := /usr/include/python3.8
PYBINDINCLUDE := `python3 -m pybind11 --includes`
PYCONFIG      := `python3-config --extension-suffix`

HDRS          := $(wildcard $(HEADERPATH)/*.hpp)
SRCS          := $(wildcard $(SOURCEPATH)/*.cpp)

TRASH         := $(EXES) *.so __pycache__ .pytest_cache save_node_test save_layer_test save_mlp_test ./test/__pycache__ test_best.mlp example_and example_va

all: utils neuron layer mlp
	@echo "compile successfully"

test: utils neuron layer mlp
	python3 -m pytest

neuron_test: utils neuron
	python3 -m pytest ./test/test_neuron.py

layer_test: utils layer
	python3 -m pytest ./test/test_layer.py

mlp_test: mlp
	python3 -m pytest ./test/test_mlp.py

utils: $(SOURCEPATH)/utils.cpp $(HEADERPATH)/utils.hpp
	$(CXX) $(CXXFLAGS) -I$(HEADERPATH) -I$(PYINCLUDE) ${PYBINDINCLUDE} $< -o _utils${PYCONFIG}

neuron: $(SOURCEPATH)/neuron.cpp $(HEADERPATH)/neuron.hpp
	$(CXX) $(CXXFLAGS) -I$(HEADERPATH) -I$(PYINCLUDE) ${PYBINDINCLUDE} $< -o _neuron${PYCONFIG}

layer: $(SOURCEPATH)/layer.cpp $(HEADERPATH)/layer.hpp
	$(CXX) $(CXXFLAGS) -I$(HEADERPATH) -I$(PYINCLUDE) ${PYBINDINCLUDE} $< -o _layer${PYCONFIG}

mlp: $(SOURCEPATH)/mlp.cpp $(HEADERPATH)/mlp.hpp
	$(CXX) $(CXXFLAGS) -I$(HEADERPATH) -I$(PYINCLUDE) ${PYBINDINCLUDE} $< -o _mlp${PYCONFIG}

dataset: $(SOURCEPATH)/dataset.cpp $(HEADERPATH)/dataset.hpp
	$(CXX) $(CXXFLAGS) -I$(HEADERPATH) -I$(PYINCLUDE) ${PYBINDINCLUDE} $< -o _dataset${PYCONFIG}

example_and: $(SOURCEPATH)/example_and.cpp $(HDRS)
	$(CXX) -O3 -std=c++17 -I$(HEADERPATH) -o $@ $<
	./$@

example_va: $(SOURCEPATH)/example_va.cpp $(HDRS)
	$(CXX) -O3 -std=c++17 -fopenmp -I$(HEADERPATH) -o $@ $<
	./$@

example_va_py: $(SOURCEPATH)/example_va.py mlp dataset
	mv *.so src
	python3 -m $(SOURCEPATH)/example_va

.PHONY: clean

clean:
	rm -rf $(TRASH) src/*.so