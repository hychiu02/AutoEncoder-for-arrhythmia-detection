#ifndef _DATASET_H_
#define _DATASET_H_

#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <filesystem>
#include <sys/stat.h>
#include <exception>

namespace fs = std::filesystem;
// Use for getting all file name with path in data director
std::vector<std::string> get_fpaths(std::string dir_path)
{
// cout << std::filesystem::current_path() << endl;
    std::vector<std::string> fpaths;
    struct stat sb;
    for(const auto & entry : std::filesystem::directory_iterator(dir_path))
    {
        // Converting the path to const char * in the
        // subsequent lines
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        const char* path = outfilename_str.c_str();

        // Testing whether the path points to a
        // non-directory or not If it does, displays path
        if (stat(path, &sb) == 0 && !(sb.st_mode & S_IFDIR))
        {
            //std ::cout << path << std::endl;
            fpaths.emplace_back(path);
        }
    }

    return fpaths;
}

class Dataset
{
public:
    // Constructor
    Dataset(std::vector<std::string> fpaths)
        :m_size(0), m_fpaths(fpaths), m_idx(888)
    {
    }
    // Use for getting number of data in this dataset
    std::size_t get_dataset_size() const
    {
        return m_size;
    }
    // Use for getting all file paths in this dataset
    const std::vector<std::string> & get_fpaths() const
    {
        return m_fpaths;
    }
    // Virtual method for different dataset
    virtual void load_data(std::size_t idx) {}
    // Use for getting current file index
    std::size_t get_idx() const
    {
        return m_idx;
    }
    // Use for getting input vector
    const std::vector<double> & get_input_vector() const
    {
        return m_input_vector;
    }
    // Use for getting input vector
    std::size_t get_input_size() const
    {
        return m_input_vector.size();
    }
    // Use for getting ground truth
    const std::vector<double> & get_output_vector() const
    {
        return m_output_vector;
    }  
    // Use for getting output size
    std::size_t get_output_size() const
    {
        return m_output_vector.size();
    }
    // Use for displaying information of this dataset
    friend std::ostream & operator<<(std::ostream & ostr, Dataset const & dataset)
    {
        ostr << "========================\n";
        ostr << "size: " << dataset.get_dataset_size() << "\n";    
        ostr << "------------------------\n";
        ostr << "File names with path: [";
        for(std::string s : dataset.get_fpaths())
        {
            ostr << s << ", ";
        }
        ostr << "]\n";
        ostr << "------------------------\n";
        ostr << "Current index: " << dataset.get_idx() << "\n";
        ostr << "------------------------\n";
        ostr << "Current Input vector: [";
        for(double element : dataset.get_input_vector())
        {
            ostr << element << ", ";
        }
        ostr << "]\n";
        ostr << "------------------------\n";
        ostr << "Current Output vector: [";
        for(double element : dataset.get_output_vector())
        {
            ostr << element << ", ";
        }
        ostr << "]\n";
        ostr << "========================\n";

        return ostr;
    }

protected:
    std::size_t m_size;
    std::vector<std::string> m_fpaths;
    std::size_t m_idx;
    std::vector<double> m_input_vector;
    std::vector<double> m_output_vector;
};

class Gate_Dataset : public Dataset
{
public:
    // Constructor
    Gate_Dataset(std::vector<std::string> fpaths)
        :Dataset(fpaths)
    {
        read_file();
        m_size = m_input_set.size();
        load_data(0);
    }

    const std::size_t get_gata_dataset_size()
    {
        return m_output_set.size();
    }
    // Use for loading data to dataset(so that mlp can fetch one data)
    void load_data(std::size_t idx) override
    {
        m_input_vector = m_input_set[idx];
        m_output_vector = m_output_set[idx];
        m_idx = idx;
    }
    // Use for reading all data from a txt file for the gate
    void read_file()
    {
        
        // reading input data from file
        std::vector<std::string> data_vector;
// std::cout << "file path: " << m_fpaths[0] << "\n";

        std::ifstream ifs(m_fpaths[0], std::ios::in);


        if(!ifs.is_open())
        {
            std::cout << "Failed to open file.\n";
            throw new std::runtime_error("Failed to open file");
        }


        std::string s;
        while(std::getline(ifs, s))
        {
            data_vector.emplace_back(s);
        }
// std::cout << "data read: ";
// for(std::string element : data_vector)
// {
//     std::cout << "[" << element << "], ";
// }
// std::cout << "\n";
        ifs.close();
        // extract output information string data
        for(std::size_t i=0; i<data_vector.size(); i++)
        {
            std::vector<double> _input_vector;
            std::vector<double> _output_vector;
            std::string::size_type begin = 0;
            std::string::size_type end = data_vector[i].find(' ');
            
            while (end != std::string::npos)
            {
                _input_vector.emplace_back(stod(data_vector[i].substr(begin, end-begin)));
                begin = end + 1;
                end = data_vector[i].find(' ', begin);
            }

            begin = data_vector[i].find(';') + 1;
            _output_vector.emplace_back(stod(data_vector[i].substr(begin)));

// std::cout << "input read: ";
// for(double element : _input_vector)
// {
//     std::cout << element << ", ";
// }
// std::cout << "\n";

// std::cout << "output read: ";
// for(double element : _output_vector)
// {
//     std::cout << element << ", ";
// }
// std::cout << "\n";
            m_input_set.emplace_back(_input_vector);
            m_output_set.emplace_back(_output_vector);
        }

    }

private:
    std::vector<std::vector<double>> m_input_set;
    std::vector<std::vector<double>> m_output_set;
};

class VA_Dataset : public Dataset
{
public:
    // Constructor
    VA_Dataset(std::vector<std::string> fpaths)
        :Dataset(fpaths)
    {
        m_size = fpaths.size();
        load_data(0);
    }
    // Use for reading just one data from a txt and loaded into dataset
    void load_data(std::size_t idx) override
    {
        if(idx != m_idx)
        {
            // reading input data from file
            m_input_vector.clear();

// std::cout << m_fpaths[idx] << std::endl;
            std::ifstream ifs(m_fpaths[idx], std::ios::in);
            if(!ifs.is_open())
            {
                std::cout << "Failed to open file.\n";
                throw new std::runtime_error("Failed to open file");
            }

            std::string s;
            while(std::getline(ifs, s))
            {
                m_input_vector.emplace_back(stod(s));
            }
// std::cout << "size of input: " << _input_vector.size() << std::endl;
            ifs.close();
            // extract output information from filename
            m_output_vector.clear();
            // extract file name without path
            std::string fname;
            std::string::size_type begin = 0;
            std::string::size_type end = m_fpaths[idx].find('/');            
            while (end != std::string::npos)
            {
                begin = end + 1;
                end = m_fpaths[idx].find('/', begin);
            }
            if(begin != m_fpaths[idx].length())
            {
                fname = m_fpaths[idx].substr(begin);
            }
            // extract class name from file name
            std::string class_name;
            begin = 0;
            end = fname.find('-');
            class_name = fname.substr(begin, end-begin);
// std::cout << "class_name: " << class_name << "\n";
            if((class_name=="VT") || (class_name=="VFb") || (class_name=="VFt"))
            {
// std::cout << "is VA" << "\n";
                m_output_vector.emplace_back(1);
            }
            else if((class_name=="AFb") || (class_name=="AFt") || (class_name=="SR") || (class_name=="SVT") || (class_name=="VPD"))
            {
// std::cout << "is not VA" << "\n";
                m_output_vector.emplace_back(0);
            }
            else
            {
                std::cout << "Unknown class name: " << class_name << "\n";
                throw new std::runtime_error("Unknown class name");
            }

            m_idx = idx;
        }
    }
};


#endif