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

	//--- Getter and Setter
		enum class Attribute { 
			INPUT_SIZE, OUTPUT_SIZE, NUMBER_OF_NEURONS, 								                 // TYPE: size_t
			ALL_NEURONS, 																                 // TYPE: std::vector<Neuron>
			LAYER_OUTPUTS, LAYER_ERRORS, ALL_NEURONS_GRADIENTS, ACCUMULATED_OUTPUTS, RECEIVED_INPUT,	 // TYPE: std::vector<double>
			LEARNING_RATE																				 // TYPE: double
		};
		template <Layer::Attribute attrib> const auto Get() const;
		template <Layer::Attribute attrib, typename T> void Set(T value);


	friend class MLP;
};




/// <summary>
/// Getter
/// </summary>
/// <exemple>
/// <code>
///		size_t inputSize = layer.Get<Layer::Attribute::OUTPUT_SIZE>()
/// </code>
/// </exemple>

template<Layer::Attribute attrib>
inline const auto Layer::Get() const
{
	if constexpr (attrib == Layer::Attribute::INPUT_SIZE) {
		return _inputSize;
	}
	else if constexpr (attrib == Layer::Attribute::OUTPUT_SIZE) {
		return _outputs.size();
	}
	else if constexpr (attrib == Layer::Attribute::NUMBER_OF_NEURONS) {
		return _neurons.size();
	}
	else if constexpr (attrib == Layer::Attribute::ALL_NEURONS) {
		return _neurons;
	}
	else if constexpr (attrib == Layer::Attribute::LAYER_OUTPUTS) {
		return _outputs;
	}
	else if constexpr (attrib == Layer::Attribute::LAYER_ERRORS) {
		std::vector<double> errors;
		for (auto& neuron : _neurons) { errors.push_back( neuron.Get<Neuron::Attribute::ERROR>() ); }
		return errors;
	}
	else if constexpr (attrib == Layer::Attribute::ALL_NEURONS_GRADIENTS) {
		std::vector<double> gradients;
		for (auto& neuron : _neurons) { gradients.push_back( neuron.Get<Neuron::Attribute::GRADIENT_DL_DU>() ); }
		return gradients;
	}
	else if constexpr (attrib == Layer::Attribute::ACCUMULATED_OUTPUTS) {
		return _accumulatedLayerOutouts;
	}
	else if constexpr (attrib == Layer::Attribute::RECEIVED_INPUT) {
		return _receivedInput;
	}
	else {
		assert(false && "cant get this attribute");
	}
}




/// <summary>
/// Setter
/// </summary>
/// <exemple>
/// <code>
///		 layer.Set<Layer::Attribute::LEARNING_RATE, double>(0.03);
/// </code>
/// </exemple>

template<Layer::Attribute attrib, typename T>
inline void Layer::Set(T value)
{
	if constexpr (attrib == Layer::Attribute::LEARNING_RATE) {
		static_assert( std::is_same_v<T, double>  &&  "[ERROR]: wrong type");
		_neuronLerningRate  =  value;
	}
	else {
		assert(false  &&  "[ERROR]: not settable attribute");
	}
}
