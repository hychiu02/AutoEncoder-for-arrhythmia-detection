#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>
#include <functional>
#include <unordered_map>
#include <exception>
#include <cmath>
#include <vector>
#include <iostream>
#include <ctime>
#include <cstdlib>

class Activation
{
public:
    Activation()
    {
        add_new_pair("linear", linear, deriv_linear);
        add_new_pair("sigmoid", sigmoid, deriv_sigmoid);
    }

    ~Activation()
    {
    }

    std::function<double(double)> get_activation_func(std::string func_name)
    {
        return activation_func_map[func_name].first;
    }

    std::function<double(double)> get_deriv_activation_func(std::string func_name)
    {
        return activation_func_map[func_name].second;
    }

    std::pair<std::function<double(double)> , std::function<double(double)>> & get_pair(std::string func_name)
    {
        return activation_func_map[func_name];
    }

    static inline double linear(double x)
    {
        return  x;
    }

    static inline double deriv_linear(double x)
    {
        return 1;
    }

    static inline double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }

    // Derivative of sigmoid function
    static inline double deriv_sigmoid(double x)
    {
        return sigmoid(x)*(1 - sigmoid(x));
    }

private:
    std::unordered_map<std::string , std::pair<std::function<double(double)> , std::function<double(double)> >> activation_func_map;

    void add_new_pair(std::string func_name, std::function<double(double)> func, std::function<double(double)> deriv_func)
    {
        if(activation_func_map.count(func_name))
        {
            throw new std::invalid_argument("Duplicated func key registry");
        }
        else
        {
            activation_func_map.insert(std::make_pair(func_name, std::make_pair(func, deriv_func)));
        }
    }

};

double mse(std::vector<double> & preds, std::vector<double> & ground_truth)
{
    if((preds.size() == ground_truth.size()) || ground_truth.size()==1)
    {
        double m_error = 0.0;

        if(ground_truth.size() == 1)
        {
            for(std::size_t i=0; i<preds.size(); i++)
            {
                double delta = ground_truth[i] - preds[i];
                m_error += delta * delta;
            }
        }
        else
        {
            for(std::size_t i=0; i<preds.size(); i++)
            {
                double delta = ground_truth[i] - preds[i];
                m_error += delta * delta;
            }
        }

        return m_error / preds.size();
    }
    else
    {
        throw new std::logic_error("Different input size between prediciton result and ground truth");
    }
}

std::vector<double> deriv_mse(std::vector<double> & preds, std::vector<double> & ground_truth)
{
    if((preds.size() == ground_truth.size()) || ground_truth.size()==1)
    {
        std::vector<double> deriv_error(preds.size(), 0); 

        // let y be a predect result, gf be a ground truth
        // let u = (gf - y)
        // C = (gf - y) ^ 2 / n,
        // dC / dy = (dC / du) * (du / dy)
        // dC / du = (2 * u / n) = 2 * (gf - y) / n
        // du / dy = -1
        // dC / dy = 2 * (gf - y) / n * (-1)
        if(ground_truth.size() == 1)
        {
            for(std::size_t i=0; i<preds.size(); i++)
            {
                deriv_error[i] = 2 * (ground_truth[0] - preds[i]) / preds.size() * (-1);
            }
        }
        else
        {
            for(std::size_t i=0; i<preds.size(); i++)
            {
                deriv_error[i] = 2 * (ground_truth[0] - preds[i]) / preds.size() * (-1);
            }
        }

        return deriv_error;
    }
    else
    {
        throw new std::logic_error("Different input size between prediciton result and ground truth");
    }
}

double gen_rand_weights()
{
    return (double) 2*rand() / double(RAND_MAX) + (-1);
}

#endif