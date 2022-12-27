#ifndef _MLP_H_
#define _MLP_H_

#include <vector>
#include <string>

#include "utils.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "dataset.hpp"

class MLP
{
public:
     // If wanting to load mlp, can use this constructor
    MLP()
        :m_num_layers(0), m_num_inputs(0), m_num_outputs(0)
    {
        m_layers.clear();
    }
    // Use this constructor when building network
    MLP(std::vector<std::size_t> num_layer_neurons, std::vector<std::string> activation_func_name, bool init_with_constant_weights = false, double constant_weight = 0.5)
    {
        if(num_layer_neurons.size() < 2)
        {
            // Should include number of inputs and at least one layer
            std::cout << "Size of number of neurons of each layer less than 2 during MLP constructor" << std::endl;
            throw new std::logic_error("Size of number of neurons of each layer less than 2 during MLP constructor");          
        }
        else if(activation_func_name.size() != num_layer_neurons.size() - 1)
        {
            // Number of activation functions should be the same as number of layers(exclude inputs, which is num_layer_neurons[0])
            std::cout << "Different size between number of layers and number of activation functions during MLP constructor" << std::endl;
            throw new std::logic_error("Different size between number of layers and number of activation functions during MLP constructor");          
        }           
        else
        {
                // For loading mlp
            // m_num_lyaer_neurons = num_layer_neurons;
            // First element of m_layer_neurons is number of inputs
            m_num_inputs = num_layer_neurons[0];
            // Last element of m_layer_neurons is number of outputs
            m_num_outputs = num_layer_neurons.back();
            // Only conider hidden layers and output layer
            m_num_layers = num_layer_neurons.size() - 1;
            // Make layers
            m_layers.resize(m_num_layers);
            // m_num_layers would be m_layer_neurons - 1
            for(std::size_t i=0; i<m_num_layers; i++)
            {
                m_layers[i] = Layer(num_layer_neurons[i], num_layer_neurons[i+1], activation_func_name[i], init_with_constant_weights, constant_weight);
            }
        }
    }
    // Destructor
    ~MLP()
    {
        m_num_layers = 0;
        m_num_inputs = 0;
        m_num_outputs = 0;
        m_layers.clear();
    }
    // Use for forward pass
    void forward(std::vector<double> & inputs)
    {
        if(inputs.size() == m_num_inputs)
        {
            std::vector<double> prev_layer_out = inputs;

            for(std::size_t i=0; i<m_num_layers; i++)
            {
                m_layers[i].forward(prev_layer_out);
                prev_layer_out = m_layers[i].get_output();
            }
        }
        else
        {
            std::cout << "Different size between inputs and m_num_inputs of MLP during forwarding: " << "\n";
            std::cout << "input_size: " << inputs.size() << "\n";
            std::cout << "m_num_inputs: " << m_num_inputs << "\n";
            throw new std::logic_error("Different size between prev_layer_out and m_num_inputs of Layer during forwarding");
        }        
    }
    // Use for getting output of output layer
    std::vector<double> get_output()
    {
        return m_layers.back().get_output();
    }
    // Use for inference one data
    std::vector<double> fit(std::vector<double> & inputs)
    {
        forward(inputs);
        
        return m_layers.back().get_output();
    }
    // Use for backpropogation
    void backward(std::vector<double> & inputs, std::vector<double> & ground_truth, double learning_rate)
    {
        if(inputs.size() == m_num_inputs && ground_truth.size() == m_num_outputs)
        {
            // First calculate output gradients of output layer
            // prev_layer_out here is output layer output
            std::vector<double> prev_layer_out = m_layers.back().get_output();
            std::vector<double> gradients = deriv_mse(prev_layer_out, ground_truth);
// double loss = 0.0;
// loss = mse(prev_layer_out, ground_truth);
// std::cout << "Total MSE loss: " << loss << "\n";
            m_layers.back().update_grad(gradients);

            // Second, update gradients of hidden layers
            for(int i=m_num_layers-2; i>=0; i--)
            {
                // Neurons in next layer
                std::vector<Neuron> next_layer_neruons = m_layers[i+1].get_neurons();
                
                gradients.resize(m_layers[i].get_num_neurons());
                // Calculate gradient of each neuron in this layer

                for(std::size_t j=0; j<m_layers[i].get_num_neurons(); j++)
                {
                    gradients[j] = 0.0;
                    for(std::size_t k=0; k<next_layer_neruons.size(); k++)
                    {
                        gradients[j] += next_layer_neruons[k].get_weights()[j] * next_layer_neruons[k].get_grad();
                    }
                }
                m_layers[i].update_grad(gradients);
            }

            // Update weights after updating gradientsW
            // pre_layer_out here is exactly outputs of previous layer
            for(std::size_t i=m_num_layers-1; i>0; i--)
            {
                prev_layer_out = m_layers[i-1].get_output();
                m_layers[i].update_weights(prev_layer_out, learning_rate);
            }
            m_layers[0].update_weights(inputs, learning_rate);
        }
        else
        {
            if(inputs.size() != m_num_inputs)
            {
                std::cout << "Different size between inputs and m_num_inputs of MLP during backwarding" << std::endl;
                throw new std::logic_error("Different size between prev_layer_out and m_num_inputs of Layer during backwarding");
            }
            else
            {
                std::cout << "Different size between ground_truth and m_num_outputs of MLP during backwarding" << std::endl;
                throw new std::logic_error("Different size between prev_layer_out and m_num_inputs of Layer during backwarding");
            }
        }
    }
    // Use for training
    void train(Dataset & train_set, Dataset & val_set, double learning_rate, std::size_t max_iterations, std::string model_name)
    {
        std::size_t stale = 0;
        double best_acc = 0.0;

        for(std::size_t i=0; i<max_iterations; i++)
        {
            double loss = 0.0;
            // train
            for(std::size_t j=0; j<train_set.get_dataset_size(); j++)
            {
                train_set.load_data(j);
                std::vector<double> input_vector = train_set.get_input_vector();
                std::vector<double> ground_truth = train_set.get_output_vector();
                std::vector<double> preds = fit(input_vector);
/*
std::cout << "inputs:       [";
for(double element : input_vector)
{
    std::cout << element << ", ";
}
std::cout << "], ";

std::cout << "ground truth: [";
for(double element : ground_truth)
{
    std::cout << element << ", ";
}
std::cout << "], ";

std::cout << "preds:        [";
for(double element : preds)
{
    std::cout << element << ", ";
}
std::cout << "]\n";
*/
                loss += mse(preds, ground_truth);
                backward(input_vector, ground_truth, learning_rate);
            }
//std::cout << "Training loss: " << loss << "\n";
            // validate
            double val_acc = test(val_set);

            if(val_acc > best_acc)
            {
                save_mlp(model_name+"_best.mlp");
                best_acc = val_acc;
                stale = 0;
            }
            else
            {
                // if(++stale > 30)
                // {
                //     std::cout << "Early exist\n";
                //     break;
                // }
            }
        }
    }
    // Use for validation
    double test(Dataset & dataset)
    {
        double acc = 0.0;
        double threshold = 0.5;

        for(std::size_t j=0; j<dataset.get_dataset_size(); j++)
        {
            dataset.load_data(j);
            std::vector<double> input_vector = dataset.get_input_vector();
            std::vector<double> ground_truth = dataset.get_output_vector();
            std::vector<double> preds = fit(input_vector);

            for(std::size_t i=0; i<preds.size(); i++)
            {
                if((preds[i]>threshold) == (ground_truth[i]>threshold))
                {
                    acc += 1;
                }
            }

            acc /= preds.size();
        }

        acc /= dataset.get_dataset_size();

        return acc;
    }

    // Use for inferencing
    std::vector<std::vector<double>> inference(Dataset & dataset)
    {
        std::vector<std::vector<double>> ret;

        for(std::size_t j=0; j<dataset.get_dataset_size(); j++)
        {
            dataset.load_data(j);
            std::vector<double> input_vector = dataset.get_input_vector();
            std::vector<double> preds = fit(input_vector);

            ret.emplace_back(preds);
        }

        return ret;
    }

    // Use for saving model
    void save_mlp(const std::string & fname) const
    {
        FILE * file;
        file = fopen(fname.c_str(), "wb");
        fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
        fwrite(&m_num_outputs, sizeof(m_num_outputs), 1, file);
        fwrite(&m_num_layers, sizeof(m_num_layers), 1, file);
        // fwrite(&m_layer_neurons[0], sizeof(m_layers_neurons[0]), m_layer_neurons.size(), file);
        for(std::size_t i=0; i<m_layers.size(); i++)
        {
            m_layers[i].save_layer(file);
        }
        fclose(file);
    }
    // Use for loading model
    void load_mlp(const std::string & fname)
    {
        FILE * file;
        file = fopen(fname.c_str(), "rb");
        fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
        fread(&m_num_outputs, sizeof(m_num_outputs), 1, file);
        fread(&m_num_layers, sizeof(m_num_layers), 1, file);
        // fwrite(&m_layer_neurons[0], sizeof(m_layers_neurons[0]), m_layer_neurons.size(), file);
        m_layers.resize(m_num_layers);
        for(std::size_t i=0; i<m_num_layers; i++)
        {
            m_layers[i].load_layer(file);
        }
        fclose(file);      
    }
    // Use for displaying information of this model
    friend std::ostream & operator<<(std::ostream & ostr, const MLP & mlp)
    {
        ostr << "num inputs: " << mlp.m_num_inputs << "\n";
        ostr << "num layers: " << mlp.m_num_layers << "\n";
        ostr << "layers: " << "\n";
        for(std::size_t i=0; i<mlp.m_layers.size(); i++)
        {
            ostr << "===== layer " << i << "=====" << "\n";
            ostr << mlp.m_layers[i] << "\n";
        }
        ostr << "\n";

        return ostr;
    }

private:
    // std::vector<std::size_t> m_num_lyaer_neurons;
    std::size_t m_num_inputs;
    std::size_t m_num_outputs;
    std::size_t m_num_layers;
    std::vector<Layer> m_layers;   
};

#endif