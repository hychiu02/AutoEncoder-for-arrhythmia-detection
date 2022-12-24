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

echo "INFO: 'make mlp' must work"
make mlp; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

echo "INFO: validate using pytest"
python3 -m pytest $test_path/test_mlp2.py; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

make clean;
exit 0
':'''
import unittest
import os

from _mlp import MLP

class Training_sample:
    def __init__(self, input_lst, output_lst):
        self.input_lst = input_lst
        self.output_lst = output_lst
    
    def __len__(self):
        return len(self.input_lst)
    
    def get_input(self):
        return self.input_lst
    
    def get_output(self):
        return self.output_lst

def make_training_set(input_lst, output_lst):
    training_sample_set = []
    for idx in range(len(input_lst)):
        training_sample_set.append(Training_sample(input_lst[idx], output_lst[idx]))
    
    return training_sample_set

def train(mlp: MLP, training_sample_set, learning_rate: float, max_iterations: int):
    print("Oringin MLP: {}".format(mlp))

    stale = 0
    threshold = 0.5
    for i in range(max_iterations):
        for training_sample in training_sample_set:
            inputs = training_sample.get_input()
            ground_truth = training_sample.get_output()

            # mlp.forward(inputs)
            # preds = mlp.get_output()
            preds = mlp.fit(inputs)

            if((preds[0]>threshold) ^ (ground_truth[0]>threshold)):
                stale = 0
                mlp.backward(inputs, ground_truth, learning_rate)
            else:
                stale += 1
        if(stale > 30):
            print("========== Early exist ==========")
            break

    print("Trained MLP: {}".format(mlp))



def test_learn_and():
    print("Train AND function with Node")

    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[0.0], [0.0], [0.0], [1.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    num_layer_neurons = [2, 2, 1]
    layer_activation_funcs = ["sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    train(mlp, training_sample_set, 0.1, 500)

    threshold = 0.5
    for training_sample in training_sample_set:
        inputs = training_sample.get_input()
        # mlp.forward(inputs)
        # pred = mlp.get_output()
        preds = mlp.fit(inputs)
        ground_truth = training_sample.get_output()

        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_nand():
    print("Train NAND function with Node")

    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[1.0], [1.0], [1.0], [0.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    num_layer_neurons = [2, 2, 1]
    layer_activation_funcs = ["sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    train(mlp, training_sample_set, 0.5, 500)

    threshold = 0.5
    for training_sample in training_sample_set:
        inputs = training_sample.get_input()
        # mlp.forward(inputs)
        # pred = mlp.get_output()
        preds = mlp.fit(inputs)
        ground_truth = training_sample.get_output()

        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_or():
    print("Train OR function with Node")

    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[0.0], [1.0], [1.0], [1.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    num_layer_neurons = [2, 2, 1]
    layer_activation_funcs = ["sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    train(mlp, training_sample_set, 0.5, 500)

    threshold = 0.5
    for training_sample in training_sample_set:
        inputs = training_sample.get_input()
        # mlp.forward(inputs)
        # pred = mlp.get_output()
        preds = mlp.fit(inputs)
        ground_truth = training_sample.get_output()

        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_nor():
    print("Train NOR function with Node")

    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[1.0], [0.0], [0.0], [0.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    num_layer_neurons = [2, 2, 1]
    layer_activation_funcs = ["sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    train(mlp, training_sample_set, 0.5, 500)

    threshold = 0.5
    for training_sample in training_sample_set:
        inputs = training_sample.get_input()
        # mlp.forward(inputs)
        # pred = mlp.get_output()
        preds = mlp.fit(inputs)
        ground_truth = training_sample.get_output()

        print("pred: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_not():
    print("Train NAND function with Node")

    input_lst = [[0], [1]]
    output_lst = [[1.0], [0.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    num_layer_neurons = [1, 1]
    layer_activation_funcs = ["linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    train(mlp, training_sample_set, 0.5, 500)

    threshold = 0.5
    for training_sample in training_sample_set:
        inputs = training_sample.get_input()
        # mlp.forward(inputs)
        # pred = mlp.get_output()
        preds = mlp.fit(inputs)
        ground_truth = training_sample.get_output()

        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_learn_xor():
    print("Train XOR function with Node")

    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[0.0], [1.0], [1.0], [0.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    num_layer_neurons = [2, 2, 1]
    layer_activation_funcs = ["sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)
    
    train(mlp, training_sample_set, 0.5, 500)

    threshold = 0.5
    for training_sample in training_sample_set:
        inputs = training_sample.get_input()
 
        preds = mlp.fit(inputs)
        ground_truth = training_sample.get_output()

        print("preds: {}, ground_truth: {}".format(preds, ground_truth))

        assert((preds[0] > threshold) == (ground_truth[0] > 0.5))
    print("Trained with success")

def test_save_load():
    origin_mlp = MLP([2, 4, 8, 4, 2, 1], ["sigmoid", "sigmoid", "sigmoid", "sigmoid", "linear"])
    origin_mlp.save_mlp("save_mlp_test")
    print("origin mlp: {}".format(origin_mlp))

    new_mlp = MLP()
    print("new mlp: {}".format(new_mlp))
    new_mlp.load_mlp("save_mlp_test")
    print("loaded mlp: {}".format(new_mlp))

    assert(origin_mlp.__str__() == new_mlp.__str__())