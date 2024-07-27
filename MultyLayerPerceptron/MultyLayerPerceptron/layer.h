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
		ILossFunction* _lostFunction;
		double _neuronLerningRate;

	//--- attribute to store infortant values
		size_t _inputSize;
		std::vector<double> _accumulatedLayerOutouts;
		std::vector<double> _receivedInput;

		double _alpha;  // scale (used in batch Norm)
		double _beta;   // shift (used in batch Norm)


	public:
	//--- construtor
		Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun = new Sigmoid(), double neuronLerningRate = 0.01, ILossFunction* _lostFunction = nullptr);
		~Layer();

	//--- main methods
		std::vector<double> CalculateLayerOutputs(std::vector<double> inputs, std::vector<double>* means = nullptr, std::vector<double>* devs = nullptr);
		std::vector<double> UpdateLastLayerNeurons(std::vector<double> correctValues, std::vector<double> predictedValues, std::vector<double> inputs, size_t* batchSize = nullptr, bool isBatchNorm = false);
		void UpdateHiddenLayerNeurons(std::vector<double> nextLayerGradient, std::vector<double> inputs, size_t* batchSize = nullptr, bool isBatchNorm = false);
		double GradientAtIndex(int index);
		std::vector<double> Gradients();								

	//--- Auxiliar methods
		std::vector<double> CalculateAccumulatedError(std::vector<double> correctValues, std::vector<double> predictedValue);
		void UpdateBatchNormParms();
		std::vector<double> MeanAccumulatedOutput(double batchSize);
		Neuron& operator[](size_t neuronIndex);
		friend std::ostream& operator<<(std::ostream& os, Layer layer);
		Json ToJson() const;
		Layer LoadWeightsFromJson(const Json& j);


	friend class MLP;
};
