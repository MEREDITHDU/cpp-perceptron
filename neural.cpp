#include <stdexcept>
#include "neural.h"
#include <math.h>
#include <random>


float rand_f() {
    static std::mt19937 engine;
    static std::uniform_real_distribution<float> distr(0.01, 1.0);
    float x = distr(engine);
    return x;
}


float dot(std::vector<float> & x, std::vector<float> & y) {
    if(x.size() != y.size())
        throw new std::out_of_range("wrong sizes!");
    float res = 0;
    for(int i = 0; i < x.size(); i++)
        res += x[i]*y[i];
    return res;
}

float fold(std::vector<float> & x) {
    float res = 0;
    for(float a: x) res += a;
    return res;
}


layer::layer(int input_size, int size) :
out(size), bias(size), bias_c(size), errors(size), w(size), w_c(size) {
    for(int i = 0; i < size; i++) {
        w[i] = *new std::vector<float>(input_size);
        w_c[i] = *new std::vector<float>(input_size);
        for(float & x : w[i]) x = rand_f();
        for(float & x : w_c[i]) x = rand_f();
    }
    for(float & x : bias) x = rand_f();
    this->size = size;
    momentum = 0.4;
    lrn_speed = 0.06;
}

neural::neural(int input_size, int output_size) :
        layers(3), input(input_size), output(output_size),
        errors(output_size) {
    int delta = (input_size - output_size )/ 5;
    layers.reserve(5);
    layers[0] = *new layer(input_size, input_size - delta);
    layers[1] = *new layer(input_size - delta, input_size - 2*delta);
    layers[2] = *new layer(input_size - 2*delta, input_size - 3*delta);
    layers[3] = *new layer(input_size - 3*delta, input_size - 4*delta);
    layers[4] = *new layer(input_size - 4*delta, output_size);
    lrn_speed = 0.06; momentum = 0.4;
}

std::vector<float> & neural::calculate(std::vector<float> input) {
    std::copy(input.begin(), input.end(), this->input.begin());
    for(int i = 0; i < layers[0].size; i++)
        layers[0].out[i] = tanh(layers[0].bias[i] + dot(input, layers[0].w[i]));
    for(int l = 1; l < layers.size(); l++) {
        layer & current = layers[l];
        layer & prev = layers[l-1];
        for(int i = 0; i < current.size; i++) {
            current.out[i] = tanh(current.bias[i] + dot(current.w[i], prev.out));
        }
    }
    std::vector<float> & fin = layers.back().out;
    std::copy(fin.begin(), fin.end(), output.begin());
    return output;
}


void update_w(
                layer & layer,
                std::vector<float> & errors,
                std::vector<float> & input)
{
    for(int i = 0; i < layer.size; i++) {
        float temp = errors[i] * (1 - layer.out[i]* layer.out[2]) * layer.lrn_speed;
        temp *= (1 - layer.momentum);
        layer.bias_c[i] *= layer.momentum;
        layer.bias_c[i] += temp;
        for(int j = 0; j < input.size(); j++) {
            layer.w_c[i][j] *= layer.momentum;
            layer.w_c[i][j] += temp*input[j];
            layer.w[i][j] += layer.w_c[i][j];
        }
        layer.bias[i] += layer.bias_c[i];
    }
}

float neural::learn(std::vector<float> &target) {
    if(target.size()!=output.size())
        throw new std::out_of_range("wrong target size");

    for(int i = 0; i < output.size(); i++)
        errors[i] = target[i] - output[i];
    update_w(layers.back(), errors, layers[layers.size() - 2].out);
    for(int i = layers.size() - 2; i>=1; i--)
        update_w(layers[i], layers[i+1].errors, layers[i-1].out);
    update_w(layers[0], layers[1].errors, input);
    return fold(errors)/2;
}

void neural::reset() {
    for(float & x : input) x = 0;
    for(float & x : output) x = 0;
    for(float & x : errors) x = 0;
    for(layer & x : layers) {
        for(float & a : x.out) a = 0;
        for(float & a : x.bias) a = rand_f();
        for(float & a : x.bias_c) a = 0;
        for(float & a : x.errors) a = 0;
        for(int i = 0; i < x.w.size(); i++) {
            int i_s = x.w[i].size();
            x.w[i].clear();
            x.w[i] = *new std::vector<float>(i_s, rand_f());
        }
        for(auto && a : x.w_c)
            for(float & c : a) c = 0;
    }
}

void neural::momentum_rate(float moment) {
    for(layer & x : layers)
        x.momentum = moment;
    momentum = moment;
}

void neural::learning_speed(float x) {
    for(layer & i : layers)
        i.lrn_speed = x;
    lrn_speed = x;
}


neural::~neural() {
    input.clear();
    output.clear();
    errors.clear();
    layers.clear();
}

layer::~layer() {
    bias.clear();
    bias_c.clear();
    errors.clear();
    out.clear();
    for(auto && x : w) x.clear();
    for(auto && x : w_c) x.clear();
    w.clear();
    w_c.clear();
}
