#pragma once

#include <limits>
#include "utils.h"
#include "neuron.h"

class MLP;


class Layer {

	private:
	//--- layer calss main attributes
		std::vector<Neuron> _neurons;
		std::vector<double> _outputs;

	//--- general information of this layer neurons
		IActivationFunction* _activationFunction;
		ILostFunction* _lostFunction;
		double _neuronLerningRate;

	//--- attribute to store infortant values
		size_t _inputSize;
		std::vector<double> _accumulatedLayerOutouts;
		std::vector<double> _receivedInput;

		double _alpha;  // scale (used in batch Norm)
		double _beta;   // shift (used in batch Norm)


	public:
	//--- construtor
		Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun = new Sigmoid(), double neuronLerningRate = 0.01, ILostFunction* _lostFunction = nullptr);
		~Layer();


	friend class MLP;
};
