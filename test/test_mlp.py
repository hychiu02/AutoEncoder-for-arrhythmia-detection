#!/bin/bash

''':'
test_path=test/
fail_msg="*** validation failed"

echo "INFO: 'make clean' must work"
make clean ; ret=$?

if [ -n "$(ls _utils*.so _mlp 2> /dev/null)" ] ; then
    echo "$fail_msg for uncleanness"
    exit 1
fi

echo "INFO: 'make dataset mlp' must work"
make dataset mlp; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

echo "INFO: validate using pytest"
python3 -m pytest $test_path/test_mlp.py; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

make clean;
exit 0
':'''
import unittest
import os

from _mlp import MLP
from _dataset import Gate_Dataset

data_dir = "./data/unit_test/"

def test_learn_and():
    print("Train AND function with Node")
    model_name = "test"
    and_path = [data_dir+"and.txt"]
    print(and_path)

    AND_set = Gate_Dataset(and_path)

    print("dataset size: {}".format(len(AND_set)))

    num_layer_neurons = [2, 1]
    layer_activation_funcs = ["linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    print(mlp)

    mlp.train(AND_set, AND_set, 0.1, 100, model_name)
    mlp.load_mlp("test_best.mlp")

    print('----------')
    print(mlp)

    threshold = 0.5
    for i in range(len(AND_set)):
        AND_set.load_data(i)
        inputs = AND_set.get_input_vector()
        ground_truth = AND_set.get_output_vector()
        preds = mlp.fit(inputs)       
        
        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_nand():
    print("Train NAND function with Node")
    model_name = "test"
    nand_path = [data_dir+"nand.txt"]
    print(nand_path)

    NAND_set = Gate_Dataset(nand_path)

    print("dataset size: {}".format(len(NAND_set)))

    num_layer_neurons = [2, 1]
    layer_activation_funcs = ["linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    print(mlp)

    mlp.train(NAND_set, NAND_set, 0.1, 100, model_name)
    mlp.load_mlp("test_best.mlp")

    print('----------')
    print(mlp)

    threshold = 0.5
    for i in range(len(NAND_set)):
        NAND_set.load_data(i)
        inputs = NAND_set.get_input_vector()
        ground_truth = NAND_set.get_output_vector()
        preds = mlp.fit(inputs)       
        
        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_or():
    print("Train OR function with Node")
    model_name = "test"
    or_path = [data_dir+"or.txt"]
    print(or_path)

    OR_set = Gate_Dataset(or_path)

    print("dataset size: {}".format(len(OR_set)))

    num_layer_neurons = [2, 1]
    layer_activation_funcs = ["linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    print(mlp)

    mlp.train(OR_set, OR_set, 0.1, 100, model_name)
    mlp.load_mlp("test_best.mlp")

    print('----------')
    print(mlp)

    threshold = 0.5
    for i in range(len(OR_set)):
        OR_set.load_data(i)
        inputs = OR_set.get_input_vector()
        ground_truth = OR_set.get_output_vector()
        preds = mlp.fit(inputs)       
        
        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_nor():
    print("Train NOR function with Node")
    model_name = "test"
    nor_path = [data_dir+"nor.txt"]
    print(nor_path)

    NOR_set = Gate_Dataset(nor_path)

    print("dataset size: {}".format(len(NOR_set)))

    num_layer_neurons = [2, 1]
    layer_activation_funcs = ["linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    print(mlp)

    mlp.train(NOR_set, NOR_set, 0.1, 100, model_name)
    mlp.load_mlp("test_best.mlp")

    print('----------')
    print(mlp)

    threshold = 0.5
    for i in range(len(NOR_set)):
        NOR_set.load_data(i)
        inputs = NOR_set.get_input_vector()
        ground_truth = NOR_set.get_output_vector()
        preds = mlp.fit(inputs)       
        
        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_not():
    print("Train NOT function with Node")
    model_name = "test"
    not_path = [data_dir+"not.txt"]
    print(not_path)

    NOT_set = Gate_Dataset(not_path)

    print("dataset size: {}".format(len(NOT_set)))

    num_layer_neurons = [1, 1]
    layer_activation_funcs = ["linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    print(mlp)

    mlp.train(NOT_set, NOT_set, 0.1, 100, model_name)
    mlp.load_mlp("test_best.mlp")

    print('----------')
    print(mlp)

    threshold = 0.5
    for i in range(len(NOT_set)):
        NOT_set.load_data(i)
        inputs = NOT_set.get_input_vector()
        ground_truth = NOT_set.get_output_vector()
        preds = mlp.fit(inputs)       
        
        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_xor():
    print("Train XOR function with Node")
    model_name = "test"
    xor_path = [data_dir+"xor.txt"]
    print(xor_path)

    XOR_set = Gate_Dataset(xor_path)

    print("dataset size: {}".format(len(XOR_set)))

    num_layer_neurons = [2, 2, 1]
    layer_activation_funcs = ["sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    print(mlp)

    mlp.train(XOR_set, XOR_set, 0.3, 500, model_name)
    mlp.load_mlp("test_best.mlp")

    print('----------')
    print(mlp)

    threshold = 0.5
    for i in range(len(XOR_set)):
        XOR_set.load_data(i)
        inputs = XOR_set.get_input_vector()
        ground_truth = XOR_set.get_output_vector()
        preds = mlp.fit(inputs)       
        
        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")