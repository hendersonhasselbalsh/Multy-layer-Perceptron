#include "neuron.h"

/// <summary>
///     Neuron class constructor,
///     initializes the weights
///     randomly.
/// </summary>
/// <param name="inputSize"> Size of the input vector. </param>
/// <param name="actFun"> Activation function; by default, it uses Sigmoid. </param>
/// <param name="learningRate"> Learning rate; by default, it is 0.01. </param>
/// <param name="lossFunc"> Loss function, only set for the neurons in the last layer; by default, it is nullptr. </param>

Neuron::Neuron(size_t inputSize, IActivationFunction* actFun, double leraningRate, ILossFunction* lossFunc)
    : _inputSize(inputSize+1), _learningRate(leraningRate), activationFunction(actFun), _lossFunction(lostFunc)
{
    _error = 0.0;
    _output = 0.0;
    _u = 0.0;
    _gradient = 0.0;
    _accumulatedU = 0.0;

    _weights = std::vector<double>(_inputSize, 1.0);

    for (auto& weight : _weights) {
        weight  =  Utils::RandomNormalDistributionValue(-0.3, 0.3);
    }
}




/// <summary>
///     Free all alocated pointers
/// </summary>
Neuron::~Neuron()
{
    delete activationFunction;
    delete _lossFunction;
}
