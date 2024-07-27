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

};

