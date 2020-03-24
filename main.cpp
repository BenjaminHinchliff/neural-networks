// MNIST.cpp : Defines the entry point for the application.

// std
#include <iostream>
#include <chrono>
#include <random>
// eigen
#include <Eigen/Dense>
// local
#include "load_mnist_dataset.h"
#include "Network.h"

namespace chrono = std::chrono;
using timer_clock = std::chrono::high_resolution_clock;

int main()
{
	//auto start = timer_clock::now();
	//MNISTDataset train_dataset = load_mnist_dataset("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
	//auto end = timer_clock::now();
	//chrono::duration<double> runtime = end - start;
	//printf("loaded %d sets of training data in %f seconds", train_dataset.num_items, runtime.count());

	//std::random_device rd;
	//std::mt19937 gen(rd());
	//std::normal_distribution<double> dis(0.0, 1.0);

	//Eigen::MatrixXd test = Eigen::MatrixXd::NullaryExpr(3, 3, [&]() { return dis(gen); });
	//std::cout << test << '\n';

	Network net = Network(std::vector<int>{2, 3, 2});
	//std::cout << "Weights\n" << net.weights << '\n';
	//std::cout << "Biases\n" << net.biases << '\n';

	net.weights[0][0] << 0.1, 0.2;
	net.weights[0][1] << 0.3, 0.4;
	net.weights[0][2] << 0.5, 0.6;
	net.weights[1][0] << 0.7, 0.8, 0.9;
	net.weights[1][1] << 1.0, 1.1, 1.2;

	net.biases[0] << 0.1, 0.2, 0.3;
	net.biases[1] << 0.4, 0.2;

	Eigen::VectorXd a(2);
	a << 0.1, 0.2;
	Eigen::VectorXd target(2);
	target << 0.3, 0.4;
	
	net.backprop(a, target);

	return 0;
}
