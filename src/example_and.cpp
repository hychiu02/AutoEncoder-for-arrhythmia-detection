#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <ctime>

#include <filesystem>
#include <sys/stat.h>
#include <fstream>
#include <cstdlib>

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

    std::string train_set_path = data_dir_path + "unit_test/";

    // Get train set
    std::string and_path = train_set_path + "xor.txt";
    std::vector<std::string> and_paths = {and_path};
    Gate_Dataset AND_set(and_paths);


// std::cout << VA_train_set << std::endl; 
    std::size_t num_inputs = AND_set.get_input_size();
    std::size_t num_outputs = AND_set.get_output_size();
    std::vector<std::size_t> num_layer_neurons {num_inputs, 2, num_outputs};
    std::vector<std::string> activation_func_name {"sigmoid", "linear"};

    MLP my_mlp = MLP(num_layer_neurons, activation_func_name, false);

    std::vector<double> inputs = AND_set.get_input_vector();
    std::vector<double> ground_truth = AND_set.get_output_vector();
    std::vector<double> preds = my_mlp.fit(inputs);
    std::string model_name = "./saved_model/mlp_and";

std::cout << AND_set << "\n";

for(std::size_t i=0; i<AND_set.get_dataset_size(); i++)
{
    AND_set.load_data(i);
    inputs = AND_set.get_input_vector();
    std::cout << "input vector: ";
    for(double element : inputs)
    {
        std::cout << element << " ";
    }
    std::cout << "\n";
    
    ground_truth = AND_set.get_output_vector();
    std::cout << "outputput vector: ";
    for(double element : ground_truth)
    {
        std::cout << element << " ";
    }
    std::cout << "\n";
}
    std::cout << "Origin acc: " << my_mlp.test(AND_set) << "\n";

    my_mlp.train(AND_set, AND_set, 0.3, 500, model_name);

    my_mlp.load_mlp(model_name+"_best.mlp");

    std::cout << "Trained acc: " << my_mlp.test(AND_set) << "\n";

    return 0;
}