#include <iostream>
#include <random>
#include <iomanip>
#include "neural.h"

int rand_i32()
{
    static std::mt19937 engine(time(NULL));
    static std::uniform_int_distribution<int> distr(0, 3);
    return distr(engine);
}

struct pair
{
    std::vector<float> in;
    std::vector<float> t;
};

// pairs that determines XOR problem

std::vector<pair> lrn_pairs
        {
                {{1,1},     {1}},
                {{1,-1},    {-1}},
                {{-1,1},    {-1}},
                {{-1,-1},   {1}}
        };

int main() {
    neural n(2,1);
    n.learning_speed(1);
    n.momentum_rate(0.4);
    for(int i = 0; i < 6000; i++)
    {
        auto p = lrn_pairs[rand_i32()];
        n.calculate(p.in);
        n.learn(p.t);
    }
    for(int i = 0; i < 4; i++)
    {
        std::cout << std::setprecision(1)  <<"input : [" << lrn_pairs[i].in[0] << "] [" << lrn_pairs[i].in[1] << "]";
        std::vector<float> & result = n.calculate(lrn_pairs[i].in);
        std::cout << std::setprecision(1)  <<"\nresult : [" << result[0] << "]";
        std::cout << std::endl <<std::endl;
    }
    return 0;
}
