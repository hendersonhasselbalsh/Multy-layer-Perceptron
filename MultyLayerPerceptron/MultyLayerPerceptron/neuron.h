// The Neuron class will be responsible for
// performing all operations to update
// weights using gradient descent.

#pragma once

#include "basic-includes.h"
#include <nlohmann/json.hpp>
#include "activation-functions.h"
#include "lost-function.h"
#include "utils.h"

using Json = nlohmann::json;


class Neuron {

	private:
	//--- main attributes for a neuron
		double _learningRate;                      
		std::vector<double> _weights;              // wieght and bias (weights[0])
		IActivationFunction* activationFunction;   
		ILossFunction* _lossFunction;              
		double _gradient;						   // gradient of loss With Respect To U (dL/dU) = (dO/dU) * (dL/dO)

	//--- variables to store important values
		size_t _inputSize;                                              
		double _error;	                                                
		double _output;	                                               
		double _u;		          // weighted sum                                      

		double _accumulatedU;


	public:
	//--- construtor
		Neuron(size_t inputSize, IActivationFunction* actFun = new Tanh(), double leraningRate = 0.03, ILossFunction* lossFunc = nullptr);
		~Neuron();

	//--- fundamental methods
		double CalculateOutput(std::vector<double> inputs);
		double CalculateError(double correctValue, double predictedValue, size_t* batchSize = nullptr);
		double CalculateGradient(double gradientLostWithRespectToOutput, size_t* batchSize = nullptr);
		void UpdateWeights(std::vector<double> receivedInputs);
		const double Gradient(size_t index);

	//--- auxiliar methods
		double& operator[](int weightIndex);
		friend std::ostream& operator<<(std::ostream& os, Neuron neuron);
		Neuron operator=(const Neuron& neuron);
		Json ToJson() const;
		std::vector<double> LoadWeightsFromJson(const Json& j);

	//--- getter and setter
		enum class Attribute { 
			BIAS, ERROR, U, GRADIENT_DL_DU, LEARNING_RATE,      // TYPE: double 
			WEIGHTS, OUTPUT, 									// TYPE: std::vector<double> 
			ACTIVATION_FUNC,									// TYPE: IActivationFunction* 
			LOST_FUNC											// TYPE: ILostFunction* 
		};
		template <Attribute attrib> const auto Get() const;
		template <Attribute attrib, typename T> void Set(T value);

};




/// <summary>
/// Getters
/// </summary>
/// <exemple>
/// <code>
///		double error = neuron.Get<Neuron::Attribute::ERROR>();
/// </code>
/// </exemple>

template<Neuron::Attribute attrib>
const auto Neuron::Get() const
{
	if constexpr (attrib == Neuron::Attribute::ACTIVATION_FUNC) {
		return activationFunction;
	}
	else if constexpr (attrib == Neuron::Attribute::LOST_FUNC) {
		return _lossFunction;
	}
	else if constexpr (attrib == Attribute::WEIGHTS) {
		return _weights;
	} 
	else if constexpr (attrib == Attribute::BIAS) {
		return _weights[0];
	} 
	else if constexpr (attrib == Attribute::ERROR) {
		return _error;
	} 
	else if constexpr (attrib == Attribute::U) {
		return _u;
	} 
	else if constexpr (attrib == Attribute::OUTPUT) {
		return _output;
	} 
	else if constexpr (attrib == Attribute::GRADIENT_DL_DU) {
		return _gradient;
	} 
	else {
		assert(false  &&  "Unsupported attribute");
	}
}




/// <summary>
/// Setters
/// </summary>
/// <exemple>
/// <code>
///		neuron.Set<Neuron::Attribute::LEARNING_RATE, double>(0.003);
/// </code>
/// </exemple>

template<Neuron::Attribute attrib, typename T>
void Neuron::Set(T value)
{
	if constexpr (attrib == Neuron::Attribute::ACTIVATION_FUNC) {
		static_assert( std::is_same_v<T, IActivationFunction*>  &&  "wrong type");
		activationFunction  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::LOST_FUNC) {
		static_assert(std::is_same_v<T, ILossFunction*>  &&  "wrong type");
		_lossFunction  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::LEARNING_RATE) {
		static_assert(std::is_same_v<T, double>  &&  "wrong type");
		_learningRate  =  value;
	}
	else if constexpr (attrib == Neuron::Attribute::GRADIENT_DL_DU) {
		static_assert(std::is_same_v<T, double>  &&  "wrong type");
		_gradient  =  value;
	}
	else {
		assert(false && "Not settable attribute");
	}
}

