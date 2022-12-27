import os
import time
import random

from _mlp import MLP
from _dataset import VA_Dataset

data_dir = "../data/"

def main():
    save_model_path = "../saved_model/"
    model_name = "example_va"

    train_set_path = data_dir + "train/"
    fnames = [os.path.join(train_set_path, fname) for fname in os.listdir(train_set_path)]

    train_set = VA_Dataset(fnames)

    val_set_path = data_dir + "val/"
    fnames = [os.path.join(val_set_path, fname) for fname in os.listdir(val_set_path)]

    val_set = VA_Dataset(fnames)

    num_layer_neurons = [1250, 512, 128, 32, 8, 1]
    layer_activation_funcs = ["sigmoid", "sigmoid", "sigmoid", "sigmoid", "linear"]

    mlp = MLP(num_layer_neurons, layer_activation_funcs, False, 0.5)

    training_time = 0.0
    inference_time = 0.0

    print("Start training")

    for i in range(5):
        start_t = time.time()
        mlp.train(train_set, val_set, 0.3, 500, model_name)
        end_t = time.time()
        training_time += end_t - start_t
    print("Training time: {}".format(training_time / 5))

    mlp.load_mlp("{}_best.mlp".format(model_name))

    acc = mlp.test(val_set)

    print('Training acc: {}'.format(acc))

    print('Start inference')
    for i in range(5):
        start_t = time.time()
        mlp.inference(train_set)
        end_t = time.time()
        inference_time += end_t - start_t
    print("Inference time: {}".format(inference_time / 5))


if __name__ == '__main__':
    random.seed(0)
    main()