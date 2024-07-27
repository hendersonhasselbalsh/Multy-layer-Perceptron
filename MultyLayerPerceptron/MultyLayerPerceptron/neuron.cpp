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




/// <summary>
/// Calculates the activation value of the neuron given an input vector.
/// </summary>
/// <param name="inputs">The input vector.</param>
/// <returns>The neuron's activation value.</returns>

double Neuron::CalculateOutput(std::vector<double> inputs)
{
    _u = Utils::WeightedSum(inputs, _weights);         
    _output = activationFunction->f(_u);

    _accumulatedU += _u;

    return _output;
}




/// <summary>
/// Calculates the error and the gradient of the loss with respect to the weighted sum.
/// </summary>
/// <param name="correctValue">The correct value from the label vector.</param>
/// <param name="predictedValue">The predicted value of the neuron.</param>
/// <param name="batchSize">The batch size used in batch training. (Optional)</param>
/// <returns>The error calculated by the loss function.</returns>

double Neuron::CalculateError(double correctValue, double predictedValue, size_t* batchSize)
{
    if (batchSize != nullptr) {  _u = _accumulatedU / (double)(*batchSize); }

    _error = _lossFunction->f(predictedValue, correctValue);

    double dL = _lostFunction->df(predictedValue, correctValue);       // (dL/da)
    double du = activationFunction->df(_u);                            // (da/dU) 
    _gradient = du * dL;                                               // (dL/dU) = (dL/da) * (da/dU)

    return _error;
}