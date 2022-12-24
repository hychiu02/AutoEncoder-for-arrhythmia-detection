#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include <string>
#include <functional>
#include <iostream>

#include "neuron.hpp"
#include "utils.hpp"

Activation activation_manager = Activation();

class Layer
{
public:
     // If wanting to load layer, can use this constructor
    Layer()
        :m_num_inputs(0), m_num_neurons(0), m_activation_func_name("No activation function")
    {
    }
    // Use this constructor when building network
    Layer(std::size_t num_inputs, std::size_t num_neurons, std::string activation_func_name, bool init_with_constant_weights = true, double constant_weight = 0.5)
        :m_num_inputs(num_inputs), m_num_neurons(num_neurons), m_activation_func_name(activation_func_name)
    {
        m_neurons.resize(num_neurons);
        for(std::size_t i=0; i<num_neurons; i++)
            m_neurons[i] = Neuron(num_inputs, init_with_constant_weights, constant_weight);
        std::pair<std::function<double(double)>, std::function<double(double)>> act_pair = activation_manager.get_pair(activation_func_name);
        m_activation_func = act_pair.first;
        m_deriv_activation_func = act_pair.second;
    }

    ~Layer()
    {
        m_num_inputs = 0;
        m_num_neurons = 0;
        m_neurons.clear();
    }
    // Use when layer want to pass result to next layer or using previous result to update weight
    std::vector<double> get_output()
    {
        std::vector<double> ret(m_num_neurons);

        for(std::size_t i=0; i<m_num_neurons; i++)
        {        
            ret[i] = m_neurons[i].get_output();
// std::cout << "get output in layer iteration: " << i << ", return val: " << ret[i] << "\n";
        }

        return ret;
    }
    // Use while building a vector to calculate derivative errors (dC/dz)
    std::size_t get_num_neurons()
    {
        return m_num_neurons;
    }
    // In MLP level, access neurons in this layer directively to get gradients and weights 
    std::vector<Neuron> & get_neurons()
    {
        return m_neurons;
    }
    // Use for forward pass
    void forward(const std::vector<double> & prev_layer_out)
    {
        if(prev_layer_out.size() == m_num_inputs)
        {
            for(std::size_t i=0; i<m_num_neurons; i++)
            {
                m_neurons[i].forward(prev_layer_out, m_activation_func);
// std::cout << "forward in layer: iteration " << i << ", neruon output: " << m_neurons[i].get_output() << "\n";
            }
        }
        else
        {
            std::cout << "Different size between prev_layer_out and m_num_inputs of Layer during forwarding" << std::endl;
            throw new std::logic_error("Different size between prev_layer_out and m_num_inputs of Layer during forwarding");
        }
    }
    // Use for backward pass
    void update_grad(const std::vector<double> & deriv_error)
    {
        if(deriv_error.size() == m_num_neurons)
        {
            for(std::size_t i=0; i<m_num_neurons; i++)
            {
// std::cout << "deriv_error["  << i << "]: " << deriv_error[i] << std::endl;            
                m_neurons[i].update_grad(m_deriv_activation_func, deriv_error[i]); 
            }
// std::cout << "update layer finished" << std::endl;
        }
        else
        {
            std::cout << "Different size between deriv_error and m_num_inputs of Layer during updating gradient" << std::endl;
            throw new std::logic_error("Different size between deriv_error and m_num_inputs of Layer during updating gradient");
        }
    }
    // Use for backward pass
    void update_weights(const std::vector<double> & prev_layer_out, double learning_rate)
    {
        if(prev_layer_out.size() == m_num_inputs)
        {
            for(std::size_t i=0; i<m_num_neurons; i++)
            {
                m_neurons[i].update_weights(prev_layer_out, learning_rate); 
            }
        }
        else
        {
            std::cout << "Different size between deriv_error and m_num_inputs of Layer during updating weights" << std::endl;
            throw new std::logic_error("Different size between deriv_error and m_num_inputs of Layer during updating weights");
        }
    }

    void save_layer(FILE * file) const
    {
        fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
        fwrite(&m_num_neurons, sizeof(m_num_neurons), 1, file);
        
        
        std::size_t str_size = m_activation_func_name.size();
        fwrite(&str_size, sizeof(std::size_t), 1, file);
        fwrite(m_activation_func_name.c_str(), sizeof(char), str_size, file);

        for(std::size_t i=0; i< m_neurons.size(); i++)
        {
            m_neurons[i].save_neuron(file);
        }
    }

    void load_layer(FILE * file)
    {
        m_neurons.clear();

        fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
        fread(&m_num_neurons, sizeof(m_num_neurons), 1, file);        

        std::size_t str_size = 0;
        fread(&str_size, sizeof(std::size_t), 1, file);
        m_activation_func_name.resize(str_size);
        fread(&(m_activation_func_name[0]), sizeof(char), str_size, file);

        std::pair<std::function<double(double)>, std::function<double(double)>> act_pair = activation_manager.get_pair(m_activation_func_name);
        m_activation_func = act_pair.first;
        m_deriv_activation_func = act_pair.second;

        m_neurons.resize(m_num_neurons);
        for(std::size_t i=0; i< m_num_neurons; i++)
        {
            m_neurons[i].load_neuron(file);
        }
    }

    friend std::ostream & operator<<(std::ostream & ostr, const Layer & layer)
    {
        ostr << "num inputs: " << layer.m_num_inputs << "\n";
        ostr << "num neurons: " << layer.m_num_neurons << "\n";
        ostr << "neurons: " << "\n";
        for(std::size_t i=0; i<layer.m_num_neurons; i++)
        {
            ostr << "===== neuron " << i << "=====" << "\n";
            ostr << layer.m_neurons[i] << "\n";
        }
        ostr << "\n";
        ostr << "activation function name: " << layer.m_activation_func_name << "\n";

        return ostr;
    }
    

private:
    // For initialize all neurons
    std::size_t m_num_inputs;
    std::size_t m_num_neurons;
    // To store all neurons in a layer
    std::vector<Neuron> m_neurons;
    // Loading layer need this information to get activation function pairs
    std::string m_activation_func_name;
    // Activation function a derivative activation function
    std::function<double(double)> m_activation_func;
    std::function<double(double)> m_deriv_activation_func;
};

#endif