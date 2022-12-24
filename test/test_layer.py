#!/bin/bash

''':'
test_path=test/
fail_msg="*** validation failed"

echo "INFO: 'make clean' must work"
make clean ; ret=$?

if [ -n "$(ls _layer*.so _utils*.so 2> /dev/null)" ] ; then
    echo "$fail_msg for uncleanness"
    exit 1
fi

echo "INFO: 'make utils layer'' must work"
make utils layer; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

echo "INFO: validate using pytest"
python3 -m pytest $test_path/test_layer.py; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

make clean;
exit 0
':'''

import unittest
import os

from _layer import Layer
    
def test_save_load():
    origin_layer = Layer(2, 2, "linear", False)
    origin_layer.test_save_layer("save_layer_test")
    print("origin layer: {}".format(origin_layer))

    new_layer = Layer()
    print("new layer: {}".format(new_layer))
    new_layer.test_load_layer("save_layer_test")
    print("loaded layer: {}".format(new_layer))

    assert(origin_layer.__str__() == new_layer.__str__())

test_save_load()