#pragma once

#include "basic-includes.h"
#include "multy-layer-perceptron.h"



class LayerSignature {

	public:
		size_t _qntNeurons;
		IActivationFunction* _activationFunction;
		ILossFunction* _lossFunction;
		double _learningRate;

		LayerSignature(size_t qntNeurons, IActivationFunction* activationFunction = new Sigmoid(), double learningRate = 0.01, ILossFunction* lostFunction = nullptr)
			: _qntNeurons(qntNeurons), _activationFunction(activationFunction), _learningRate(learningRate), _lossFunction(lostFunction)
		{ }

};




class MlpBuilder {

	private:
		size_t inputSize;
		MLP _mlp;

	public:
		MlpBuilder();

		MLP Build();

		MlpBuilder InputSize( size_t size );
		MlpBuilder Architecture( std::vector<size_t> neuronsInLayer );
		MlpBuilder Architecture( std::vector<LayerSignature> layerSignature );
		MlpBuilder LostFunction( ILossFunction* lostFunction );
		MlpBuilder MaxEpochs( size_t epochs );
		MlpBuilder AcceptableAccuracy( double accuracy );
		MlpBuilder ParseLabelToVector( std::function<std::vector<double>(size_t)> CallBack );
		MlpBuilder SaveOn(std::string outFile);
		MlpBuilder LoadArchitectureFromJson(std::string file);
		MlpBuilder WhenToUpdateLearningRate(std::function<bool(size_t, double)> Conddition);
		MlpBuilder HowToUpdateLearningRate(std::function<double(size_t, double, double)> func);

};

