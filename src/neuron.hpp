#ifndef _NEURON_H_
#define _NEURON_H_

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <functional>
#include <exception>
#include <fstream>

#include "utils.hpp"

class Neuron
{
public:
    // If wanting to load neuron, can use this constructor
    Neuron()
        : m_num_inputs(0), m_bias(0), m_inner_prod(0), m_output(0), m_grad(0)
    {
        m_weights.clear();
    }
    // Copy constructor
    Neuron(Neuron const & other)
        : m_num_inputs(other.m_num_inputs), m_weights(other.m_weights), m_bias(other.m_bias), m_inner_prod(0), m_output(other.m_output), m_grad(other.m_grad)
    {
    }
    // Copy
    Neuron & operator=(Neuron const & other)
    {
        if(this != &other)
        {
            m_num_inputs = other.m_num_inputs;
            m_weights = other.m_weights;
            m_bias = other.m_bias;
            m_inner_prod = other.m_inner_prod;
            m_output = other.m_output;
            m_grad = other.m_grad;
        }

        return *this;
    }
    // Move
    Neuron & operator=(Neuron && other)
    {
        if(this != &other)
        {
            std::swap(m_num_inputs, other.m_num_inputs);
            m_weights.swap(other.m_weights);
            std::swap(m_bias, other.m_bias);
            std::swap(m_inner_prod, other.m_inner_prod);
            std::swap(m_output, other.m_output);
            std::swap(m_grad, other.m_grad);
        }

        return *this;
    }    
    // Use this constructor when building network
    Neuron(std::size_t num_inputs, bool init_with_constant_weights=true, double constant_weight=0.5)
        : m_num_inputs(num_inputs), m_bias(0), m_inner_prod(0), m_output(0), m_grad(0)
    {
        if(init_with_constant_weights)
            m_weights = std::vector<double>(num_inputs, 0.5);
        else
        {
            m_weights.resize(num_inputs);

            std::generate(m_weights.begin(), m_weights.end(), gen_rand_weights);
        }

// std::cout << "Initialize with weight: \n";
// for(double element : m_weights)
// {
//     std::cout << element << " ";
// }
// std::cout << "\n";
    }
    // Destructor
    ~Neuron()
    {
        m_num_inputs = 0;
        m_weights.clear();
        m_bias = 0;
        m_inner_prod = 0;
        m_output = 0;
        m_grad = 0;
        
    }
    // Use when layer want to pass result to next layer or using previous result to update weight
    double get_output() const
    {
        return m_output;
    }
    // Use for forward pass
    void forward(const std::vector<double> & prev_layer_out, std::function<double(double)> & activation_func)
    {
        if(prev_layer_out.size() == m_num_inputs)
        {
            m_inner_prod = 0.0;
            m_output = 0.0;
            // Naive inner prod

            for(std::size_t i=0; i<m_num_inputs; i++)
            {
                m_inner_prod += m_weights[i] * prev_layer_out[i];
            }

            m_inner_prod += m_bias;

            m_output = activation_func(m_inner_prod);   
        }
        else
        {
            std::cout << "Different input size between prev_layer_out and m_num_inputs of Neuron during forwarding\n";
            throw new std::logic_error("Different input size between prev_layer_out and m_num_inputs of Neuron during forwarding");
        }
    }
    // Use when prev layer require layer's gradient while backward pass (dC/dz')
    double get_grad() const
    {
        return m_grad;
    }
    // Use for backward pass
    void update_grad(std::function<double(double)> & deriv_activation_func, double deriv_error)
    {
        // dC/dz = da/dz * dC/da
        // da/dz = deriv_act_func(z)
        // dC/da -> deriv_error
        m_grad = deriv_activation_func(m_inner_prod) * deriv_error;
    }
    // Use when prev layer require layer's gradient(dz'/da)
    const std::vector<double> & get_weights() const 
    {
        return m_weights;
    }
    // Use for backward pass
    void update_weights(const std::vector<double> & prev_layer_out, double learning_rate)
    {
        if(prev_layer_out.size() == m_num_inputs)
        {

            for(std::size_t i=0; i<m_num_inputs; i++)
            {
                m_weights[i] -= prev_layer_out[i] * learning_rate * m_grad;
            }

            m_bias -= learning_rate * m_grad;
        }
        else
        {
            throw new std::logic_error("Different input size between prev_layer_out and m_num_inputs of Neuron during updating weights");
        }

// std::cout << "Updated weights: \n";
// for(double element : m_weights)
// {
//     std::cout << element << " ";
// }
// std::cout << "\n";
    }
    // Use for saving model
    void save_neuron(FILE * file) const
    {
        fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
        fwrite(&m_weights[0], sizeof(m_weights[0]), m_weights.size(),file);
        fwrite(&m_bias, sizeof(m_bias), 1, file);
    }
    // Use for loading model
    void load_neuron(FILE * file)
    {
        m_weights.clear();

        fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
        m_weights.resize(m_num_inputs);
        fread(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file);
        fread(&m_bias, sizeof(m_bias), 1, file);
    }
    // Use for displaying information about this neuron
    friend std::ostream & operator<<(std::ostream & ostr, const Neuron & neuron)
    {
        ostr << "num inputs: " << neuron.m_num_inputs << "\n";
        ostr << "weights: ";
        for(std::size_t i=0; i<neuron.m_num_inputs; i++)
        {
            ostr << neuron.m_weights[i] << " ";
        }
        ostr << "\n";
        ostr << "bias: " << neuron.m_bias << "\n";

        return ostr;
    }

private:
    // Loading neuron need this information to load weights 
    std::size_t m_num_inputs;
    // Weights and bias for both forward and backward pass
    std::vector<double> m_weights;
    double m_bias;
    // Result before activation function, for backward getting derivative activation function output (da/dz = act'(z))
    double m_inner_prod;
    // Use variable to record can avoiding forward every time while backard pass
    double m_output;
    // Use when prev layer require layer's gradient while backward pass (dC/dz')
    double m_grad;
};

#endif