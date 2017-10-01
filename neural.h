#ifndef NN_NEURAL_H
#define NN_NEURAL_H

#include <vector>

struct layer
{
    std::vector<float> out, bias, bias_c, errors;
    std::vector<std::vector<float>> w, w_c;

    int size;
    float lrn_speed, momentum;

    layer(){}

    layer(int input_size, int size);

    ~layer();
};

class neural {
protected:
    std::vector<layer> layers;
    std::vector<float> input, output, errors;

    float lrn_speed, momentum;
public:

    neural(int input_size, int output_size);

    std::vector<float> & calculate(std::vector<float> input);

    float learn(std::vector<float> & target);

    void learning_speed(float x);
    void momentum_rate(float x);

    void reset();

    ~neural();

};


#endif