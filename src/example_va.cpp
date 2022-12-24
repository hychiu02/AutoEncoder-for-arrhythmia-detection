#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <ctime>

#include <chrono>


#include "neuron.hpp"
#include "utils.hpp"
#include "layer.hpp"
#include "mlp.hpp"
#include "dataset.hpp"

int main(int argc, char * argv[])
{   
    srand(time(NULL));

    std::string data_dir_path;

    if(argc == 1)
    {
        data_dir_path = "./data/";
    }
    else if(argc == 2)
    {
        data_dir_path = argv[1];
    }
    else
    {
        std::cout << "Number of arguments should < 3, but got " << argc << std::endl;
        exit(1);
    }

    std::string train_set_path = data_dir_path + "train";
    std::string val_set_path = data_dir_path + "val";

    // Get train set
    std::vector<std::string> file_paths = get_fpaths(train_set_path);
    VA_Dataset VA_train_set(file_paths);
    // Get val set
    file_paths = get_fpaths(val_set_path);
    VA_Dataset VA_val_set(file_paths);

    std::cout << VA_train_set << std::endl; 
    std::size_t num_inputs = VA_train_set.get_input_size();
    std::size_t num_outputs = VA_train_set.get_output_size();
    std::vector<std::size_t> num_layer_neurons {num_inputs, 512, 128, 32, 8, num_outputs};
    std::vector<std::string> activation_func_name {"sigmoid", "sigmoid", "sigmoid", "sigmoid","linear"};

    MLP my_mlp = MLP(num_layer_neurons, activation_func_name, false);

    std::vector<double> inputs = VA_train_set.get_input_vector();
    std::vector<double> ground_truth = VA_train_set.get_output_vector();
    std::vector<double> preds = my_mlp.fit(inputs);
    std::string model_name = "./saved_model/autoencoder_va";

// std::cout << "input vector: ";
// for(double element : inputs)
// {
//     std::cout << element << " ";
// }
// std::cout << "\n";

    std::cout << "Origin acc: " << my_mlp.test(VA_val_set) << "\n";

    double training_time = 0.0;
    double inference_time = 0.0;
    double acc = 0.0;
    std::cout << "Start training\n";
    for(std::size_t i=0; i<5; i++)
    {
        auto start = std::chrono::steady_clock::now();
        my_mlp.train(VA_train_set, VA_val_set, 0.05, 500, model_name);
        auto end = std::chrono::steady_clock::now();
        training_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::cout << "Training time: " << training_time / 1000000 / 5 << "\n";

    my_mlp.load_mlp(model_name+"_best.mlp");

    acc = my_mlp.test(VA_val_set);
    std::cout << "Trained acc: " << acc << "\n";

    std::cout << "Start inferencing\n";
    for(std::size_t i=0; i<5; i++)
    {
        auto start = std::chrono::steady_clock::now();
        my_mlp.inference(VA_train_set);
        auto end = std::chrono::steady_clock::now();
        inference_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "Inference time: " << inference_time / 1000000 / 5<< "\n";

    

    return 0;
}