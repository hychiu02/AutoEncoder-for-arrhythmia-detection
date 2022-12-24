###### tags: `nsd22_au`
# AutoEncoder for arrhythmia detection

## Basic information
Simple, lightweight MultiLayer Perceptron(MLP) framework
Arrhythmia dectection using Intracardiac electrograms(IEGMs) signal
![](https://i.imgur.com/ZNDBBx6.png)

## Problem to solve
Heart diseases are one of the significant reasons for death all over the planet. To perform arrhythmia detection, Electrocardiograms(ECGs) and IEGMs are used to monitor hte functioning of heart by capturing electrical activity. Because of needs of real-time inference, the model sizeand  parameters should be small, autoencoder seems to be a good choice. I want to develop a simple ML framework to build, train and inference a MLP model with C++. 

There are two main algorithm to implement:
- Feed Forward
    - Every neuron will perform $z=\sum^n_{i=1}{w_ix_i + b}$
    - Sum of each neuron will pass to an activation $\sigma$ and the sum of neurons within this layer will be the input of next layer
- Backpropogation 
    - weights of neurons update by $w_i = wi-\eta\frac{\partial C}{\partial w_i}$
    - $\frac{\partial C}{\partial w_i}$ can be efficiently computed from output layer to input layer through chain rule  $\frac{\partial C}{\partial w_i} = \frac{\partial z}{\partial w_i} \frac{\partial a}{\partial z} \frac{\partial C}{\partial a}$
        $\frac{\partial z}{\partial w_i}$ is equal to the output of previous layer
        $\frac{\partial a}{\partial z}$ is value of devirative activation function
        for $\frac{\partial C}{\partial a}$, we considered two cases
        - neuron in output layer
        $\frac{\partial C}{\partial a}$ take the MSE loss($C = \frac{1}{n}\sum^n_1(y-a)^2$) which used in this project as an example, let u be `y - a`, y is ground truth of this neuron, and $x_i$ be the output of prevois layer(input to this layer and weighted by $w_i$)
        $\frac{\partial C}{\partial a} = \frac{\partial u}{\partial a} \frac{\partial C}{\partial u}, \frac{\partial u}{\partial a} is -1, \frac{\partial C}{\partial u} is \frac{2}{n}(y-a)$
        so we can easly calculate gradient of each neuron in output layer by
        $\frac{\partial C}{\partial w_i} = x_i {\sigma}^{'}(z)(-1)(\frac{2}{n}(y-a))$
        - neuron in hidden layer
        $\frac{\partial C}{\partial a}$ is weight sum of gradient of next layer  
        $\frac{\partial C}{\partial w_i} = x_i {\sigma}^{'}(z)\sum^m_1w_{ji}\frac{\partial C}{\partial z_j}$

## Prospective Users
People who want to develop a simple MLP model for abnormal sound detection, cardiac diseases classification or other regression or classification problems using a lightweight C++ library.

## System Architecture
**This simple firmwork is currently tested on Ubuntu 20.04**
There will be a Node class for storing weight, a Layer class with Nodes for forward and backward weights calculation and a MLP class containing Layers  

![](https://i.imgur.com/Q72iF3y.png)



The flow will look like this
![](https://i.imgur.com/4mRhLqb.png)



## API Description
- Neuron class
```c++
class Neuron
{
public:
    // If wanting to load neuron, can use this constructor
    Neuron();
    // Copy constructor
    Neuron(Neuron const &);
    // Copy
    Neuron & operator=(Neuron const &);
    // Move
    Neuron & operator=(Neuron &&);
    // Use this constructor when building network
    Neuron(std::size_t, bool, double);
    // Destructor
    ~Neuron();
    // Use when layer want to pass result to next layer or using previous result to update weight
    double get_output() const;
    // Use for forward pass
    void forward(const std::vector<double>, std::function<double(double)> &);
    // Use when prev layer require layer's gradient while backward pass (dC/dz')
    double get_grad() const;
    // Use for backward pass
    void update_grad(std::function<double(double)> &, double);
    // Use when prev layer require layer's gradient(dz'/da)
    const std::vector<double> & get_weights() const;
    // Use for backward pass
    void update_weights(const std::vector<double> &, double);
    // Use for saving model
    void save_neuron(FILE) const;
    // Use for loading model
    void load_neuron(FILE * file);
    // Use for displaying information about this neuron
    friend std::ostream & operator<<(std::ostream &, const Neuron &);
};
```
- Layer class
```c++
class Layer
{
public:
     // If wanting to load layer, can use this constructor
    Layer();
    // Use this constructor when building network
    Layer(std::size_t, std::size_t, std::string, bool, double);
    // Destructor
    ~Layer();
    // Use when layer want to pass result to next layer or using previous result to update weight
    std::vector<double> get_output();
    // Use while building a vector to calculate derivative errors (dC/dz)
    std::size_t get_num_neurons();
    // In MLP level, access neurons in this layer directively to get gradients and weights 
    std::vector<Neuron> & get_neurons();
    // Use for forward pass
    void forward(const std::vector<double> &);
    // Use for backward pass
    void update_grad(const std::vector<double> &);
    // Use for backward pass
    void update_weights(const std::vector<double> & , double);
    // Use for saving model
    void save_layer(FILE * file) const;
    // Use for loading model
    void load_layer(FILE * file);
    // Use for displaying information about this layer
    friend std::ostream & operator<<(std::ostream &, const Layer &);
};
```
- MLP class
```c++
class MLP
{
public:
     // If wanting to load mlp, can use this constructor
    MLP();
    // Use this constructor when building network
    MLP(std::vector<std::size_t>, std::vector<std::string>, bool, double);
    // Destructor
    ~MLP();
    // Use for forward pass
    void forward(std::vector<double> &);
    // Use for getting output of output layer
    std::vector<double> get_output();
    // Use for inference one data
    std::vector<double> fit(std::vector<double> &);
    // Use for backpropogation
    void backward(std::vector<double> &, std::vector<double> &, double);
    // Use for training
    void train(Dataset &, Dataset &, double, std::size_t, std::string);
    // Use for validation
    double test(Dataset &);
    // Use for inferencing
    std::vector<std::vector<double>> inference(Dataset &);
    // Use for saving model
    void save_mlp(const std::string &) const;
    // Use for loading model
    void load_mlp(const std::string &);
    // Use for displaying information of this model
    friend std::ostream & operator<<(std::ostream &, const MLP &)
};
```

- Dataset class
```c++
class Dataset
{
public:
    // Constructor
    Dataset(std::vector<std::string>);
    // Use for getting number of data in this dataset
    std::size_t get_dataset_size() const;
    // Use for getting all file paths in this dataset
    const std::vector<std::string> & get_fpaths() const;
    // Virtual method for different dataset
    virtual void load_data(std::size_t) {}
    // Use for getting current file index
    std::size_t get_idx() const;
    // Use for getting input vector
    const std::vector<double> & get_input_vector() const;
    // Use for getting input size
    std::size_t get_input_size() const;
    // Use for getting ground truth
    const std::vector<double> & get_output_vector() const;
    // Use for getting output size
    std::size_t get_output_size() const;
    // Use for displaying information of this dataset
    friend std::ostream & operator<<(std::ostream &, Dataset const &)
};
```
- Gate_Dataset class
```c++

class Gate_Dataset : public Dataset
{
public:
    // Constructor
    Gate_Dataset(std::vector<std::string>);
    // Use for loading data to dataset(so that mlp can fetch one data)
    void load_data(std::size_t) override;
    // Use for reading all data from a txt file for the gate
    void read_file();
};
```
- VA_Dataset class
```c++
class VA_Dataset : public Dataset
{
public:
    // Constructor
    VA_Dataset(std::vector<std::string>);
    // Use for reading just one data from a txt and loaded into dataset
    void load_data(std::size_t idx) override;
};
```
- Activation class
``` c++
class Activation
{
public:
    // Constructor
    Activation();
    // Destructor
    ~Activation();
    // Use for getting activation function
    std::function<double(double)> get_activation_func(std::string func_name)
    {
        return activation_func_map[func_name].first;
    }
    // Use for getting derivative activation function
    std::function<double(double)> get_deriv_activation_func(std::string);rn activation_func_map[func_name].second;
    }
    // Use for getting activation, derivative activation function pair
    std::pair<std::function<double(double)> , std::function<double(double)>> & get_pair(std::string);
    // Linear function
    static inline double linear(double);
    // Derivative of linear function
    static inline double deriv_linear(double);
    // Sigmoid function
    static inline double sigmoid(double);
    // Derivative of sigmoid function
    static inline double deriv_sigmoid(double);

private:
    // Add new activation, derivative activation function pair
    void add_new_pair(std::string, std::function<double(double)>, std::function<double(double)>);
};
```
- helper function
```c++
// Use for getting all file name with path in data director
std::vector<std::string> get_fpaths(std::string);
// Mean Square Error loss function
double mse(std::vector<double> &, std::vector<double> &)
// Derivative Mean Square Error loss function
std::vector<double> deriv_mse(std::vector<double> &, std::vector<double> &)
```


## Engineering Infrastructure
1. Automatic build system and how to build your program: GNU make
2. Version control: Git
3. Testing framework: Pytest
4. Wrapping C++ file: Pybind11

## Test Codes
Test codes are in `test` folder
Test files are in polyglot scripts
For `test_neuron.py`
In `Ubuntu 20.04` you can use `./test/test_neuron` to run the script and testing automaticly
Or Using `make neuron_test` command to do unit test

## Example
Using command `make example_va` can execute example `example_va.cpp` to train a Autoencoder on a small IEGM data set then save it
Basically do the following things
- Read data
- Configure MLP
- Train MLP on the dataset
- Load the model with best weights
- Test accuracy
```C++
// ... include header

int main(int argc, char * argv[])
{   
    ...
   
    std::string train_set_path = data_dir_path + "train";
    std::string val_set_path = data_dir_path + "val";

    // Get train set
    std::vector<std::string> file_paths = get_fpaths(train_set_path);
    VA_Dataset VA_train_set(file_paths);
    // Get val set
    file_paths = get_fpaths(val_set_path);
    VA_Dataset VA_val_set(file_paths);

    
    std::size_t num_inputs = VA_train_set.get_input_size();
    std::size_t num_outputs = VA_train_set.get_output_size();
    std::vector<std::size_t> num_layer_neurons {num_inputs, 512, 128, 32, 8, num_outputs};
    std::vector<std::string> activation_func_name {"sigmoid", "sigmoid", "sigmoid", "sigmoid","linear"};

    MLP my_mlp = MLP(num_layer_neurons, activation_func_name, false);

    std::vector<double> inputs = VA_train_set.get_input_vector();
    std::vector<double> ground_truth = VA_train_set.get_output_vector();
    std::vector<double> preds = my_mlp.fit(inputs);
    std::string model_name = "./saved_model/autoencoder_va";

    ...

    my_mlp.train(VA_train_set, VA_val_set, 0.05, 100, model_name);
    
    ...

    my_mlp.load_mlp(model_name+"_best.mlp");

    ...

    std::cout << "Trained acc: " << acc << "\n";

    return 0;
}
```

## To improve 

## References
- [Neural Networks, Multilayer Perceptron and the Backpropagation Algorithm](https://medium.com/@tiago.tmleite/neural-networks-multilayer-perceptron-and-the-backpropagation-algorithm-a5cd5b904fde)
- [Understanding Backpropagation Algorithm](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)
- [Make a Neural Net Simulator in C++](https://www.millermattson.com/dave/?p=54)
- [MLP (github repo by davidalbertonogueira
)](https://github.com/davidalbertonogueira/MLP)
