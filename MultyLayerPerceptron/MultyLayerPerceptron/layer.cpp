#include "layer.h"



Layer::Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun, double neuronLerningRate, ILostFunction* _lostFunction)
	: _inputSize(inputSize), _activationFunction(actFun), _neuronLerningRate(neuronLerningRate)
{
	_neurons = std::vector<Neuron>(neuronQuantity, Neuron(inputSize, actFun, neuronLerningRate, _lostFunction));

	size_t layerOutputSize = _neurons.size() + 1;
	_accumulatedLayerOutouts =  std::vector<double>(layerOutputSize, 0.0);

	_alpha = 1.0;
	_beta  = 0.0;
}



Layer::~Layer() 
{
	delete _activationFunction;
}


