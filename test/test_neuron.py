#!/bin/bash

''':'
test_path=test/
fail_msg="*** validation failed"

echo "INFO: 'make clean' must work"
make clean ; ret=$?

if [ -n "$(_utils*.so ls _neuron*.so 2> /dev/null)" ] ; then
    echo "$fail_msg for uncleanness"
    exit 1
fi

echo "INFO: 'make utils neuron' must work"
make utils neuron; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

echo "INFO: validate using pytest"
python3 -m pytest $test_path/test_neuron.py; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

make clean;
exit 0
':'''

import unittest
import os

from _neuron import Neuron
from _utils import Activation, mse, deriv_mse

am = Activation()

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

def train(
    neuron: Neuron,
    training_sample_set,
    learning_rate: float,
    max_iterations: int,
    init_with_constant_weights: bool = False,
    constant_weight: float = 0.5
):
    print("Initial Neuron: {}".format(neuron))
    threshold = 0.5
    stale = 0
    for i in range(max_iterations):
        loss = 0
        #print("========== iteration {} ==========".format(i))
        for training_sample in training_sample_set: 
            neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
            pred = neuron.get_output()

            ground_truth = training_sample.get_output()[0]

            loss += mse([pred], [ground_truth])
            if((pred>threshold) ^ (ground_truth>threshold)):
                stale = 0
                deriv_error = deriv_mse([pred], [ground_truth])
                neuron.update_grad(am.get_deriv_activation_func("linear"), deriv_error[0])
                neuron.update_weights(training_sample.get_input(), learning_rate)
            else:
                stale += 1
        if(stale > 30):
            break
    print("Trained Node inside train: {}".format(neuron))

def test_learn_and():
    print("===============================")
    print("Train AND function with Neuron")
    
    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[0.0], [0.0], [0.0], [1.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    print("Test constant inital weights")
    neuron = Neuron(2)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("-------------------------------")
    print("Test random inital weights")
    neuron = Neuron(2, False)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("===============================\n")

def test_learn_nand():
    print("===============================")
    print("Train NAND function with Neuron")
    
    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[1.0], [1.0], [1.0], [0.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    print("Test constant inital weights")
    neuron = Neuron(2)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("-------------------------------")
    print("Test random inital weights")
    neuron = Neuron(2, False)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("===============================\n")
    print("Trained with success")

def test_learn_or():
    print("===============================")
    print("Train OR function with Neuron")
    
    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[0.0], [1.0], [1.0], [1.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    print("Test constant inital weights")
    neuron = Neuron(2)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("-------------------------------")
    print("Test random inital weights")
    neuron = Neuron(2, False)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("===============================\n")
    print("Trained with success")

def test_learn_nor():
    print("===============================")
    print("Train NOR function with Neuron")
    
    input_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output_lst = [[1.0], [0.0], [0.0], [0.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    print("Test constant inital weights")
    neuron = Neuron(2)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("-------------------------------")
    print("Test random inital weights")
    neuron = Neuron(2, False)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("===============================\n")
    print("Trained with success")
    
def test_learn_not():
    print("===============================")
    print("Train NAND function with Neuron")
    
    input_lst = [[0], [1]]
    output_lst = [[1.0], [1.0]]

    training_sample_set = make_training_set(input_lst, output_lst)

    print("Test constant inital weights")
    neuron = Neuron(1)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("-------------------------------")
    print("Test random inital weights")
    neuron = Neuron(1, False)
    
    train(neuron, training_sample_set, 0.1, 100)

    threshold = 0.5
    for training_sample in training_sample_set:
        neuron.forward(training_sample.get_input(), am.get_activation_func("linear"))
        pred = neuron.get_output() > threshold

        ground_truth = training_sample.get_output()[0] > threshold

        assert(pred == ground_truth)
    print("Trained with success")
    print("===============================\n")
    print("Trained with success")

def test_save_load():
    print("===============================")
    print("Save and load Neuron")
    origin_neuron = Neuron(2, False)
    origin_neuron.test_save_neuron("save_node_test")
    print("origin neuron: {}".format(origin_neuron))

    new_neuron = Neuron()
    print("new neuron: {}".format(new_neuron))
    new_neuron.test_load_neuron("save_node_test")
    print("loaded neuron: {}".format(new_neuron))

    assert(origin_neuron.__str__() == new_neuron.__str__())
    print("===============================\n")
    