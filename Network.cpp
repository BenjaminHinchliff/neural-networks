#include "Network.h"

double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_prime(double z)
{
    return sigmoid(z) * (1.0 - sigmoid(z));
}

Eigen::VectorXd cost_derivative(const Eigen::VectorXd& output_activs, const Eigen::VectorXd& target)
{
    return (output_activs - target);
}

Network::Network()
{
}

Network::Network(std::vector<int> sizes)
    : sizes(sizes)
{
    // random setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);
    // generate random numbers with mean 0 and standard deviation 1
    auto normal_gen = [&]() { return dis(gen); };
    // set biases
    for (size_t i = 1; i < sizes.size(); ++i) {
        biases.push_back(Eigen::VectorXd::NullaryExpr(sizes[i], normal_gen));
    }
    // set weights
    for (size_t i = 1; i < sizes.size(); ++i) {
        std::vector<Eigen::VectorXd> vectors;
        for (int j = 0; j < sizes[i]; ++j) {
            vectors.push_back(Eigen::VectorXd::NullaryExpr(sizes[i - 1], normal_gen));
        }
        weights.push_back(vectors);
    }
}

Eigen::VectorXd Network::feedforward(const Eigen::VectorXd& a)
{
    Eigen::VectorXd ai = a;
    for (size_t j = 0; j < weights.size(); ++j)
    {
        Eigen::VectorXd ao(weights[j].size());
        for (size_t i = 0; i < weights[j].size(); ++i)
        {
            ao[i] = sigmoid(weights[j][i].dot(ai) + biases[j][i]);
        }
        ai = ao;
    }
    return ai;
}

void Network::backprop(const Eigen::VectorXd& a, const Eigen::VectorXd& target)
{
    // initialize nablas with 0s in the shape of bias and vectors respectively
    BiasVector nabla_b;
    for (const auto& bias : biases)
    {
        nabla_b.push_back(Eigen::VectorXd::Zero(bias.size()));
    }
    WeightVector nabla_w;
    for (const auto& weight_set : weights)
    {
        std::vector<Eigen::VectorXd> weight;
        for (const auto& origin_weight : weight_set)
        {
            weight.push_back(Eigen::VectorXd::Zero(origin_weight.size()));
        }
        nabla_w.push_back(weight);
    }

    std::vector<Eigen::VectorXd> activations{ a };
    std::vector<Eigen::VectorXd> z_primes{};
    Eigen::VectorXd cur_activation(a);
    for (size_t j = 0; j < weights.size(); ++j)
    {
        Eigen::VectorXd zp(weights[j].size());
        Eigen::VectorXd ao(weights[j].size());
        for (size_t i = 0; i < weights[j].size(); ++i)
        {
            double z = weights[j][i].dot(cur_activation) + biases[j][i];
            zp[i] = sigmoid_prime(z);
            ao[i] = sigmoid(z);
        }
        cur_activation = ao;
        z_primes.push_back(zp);
        activations.push_back(ao);
    }
    
    Eigen::VectorXd delta(cost_derivative(activations[activations.size() - 1], target).array() * z_primes[z_primes.size() - 1].array());
    nabla_b[nabla_b.size() - 1] = delta;
    std::vector<Eigen::VectorXd> first_nabla_w;
    for (int i = 0; i < delta.size(); ++i) {
        first_nabla_w.push_back(delta[i] * activations[activations.size() - 2].transpose());
    }
    nabla_w[nabla_w.size() - 1] = first_nabla_w;
    std::cout << nabla_w[nabla_w.size() - 1] << '\n';
}