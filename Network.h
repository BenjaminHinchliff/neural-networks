#ifndef NETWORK_H
#define NETWORK_H

// std
#include <iostream>
#include <vector>
#include <random>
// eigen
#include <Eigen/Dense>

// why doesn't std::vector have an ostream overload? I mean, it would be this easy! shut up this is just to debug stuff
template<typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T> in)
{
	out << '[';
	for (size_t i = 0; i < in.size() - 1; ++i) {
		out << in[i] << ", ";
	}
	out << in[in.size() - 1] << ']';
	return out;
}

using BiasVector = std::vector<Eigen::VectorXd>;
using WeightVector = std::vector<std::vector<Eigen::VectorXd>>;;

class Network
{
public:
	Network();
	Network(std::vector<int> sizes);

	Eigen::VectorXd feedforward(const Eigen::VectorXd& a);
	void backprop(const Eigen::VectorXd& a, const Eigen::VectorXd& target);
public:
	int layers = 0;
	std::vector<int> sizes = {};
	BiasVector biases = {};
	WeightVector weights = {};
};

double sigmoid(double z);
double sigmoid_prime(double z);

#endif // !NETWORK_H
