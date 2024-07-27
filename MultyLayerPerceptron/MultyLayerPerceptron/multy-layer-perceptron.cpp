#pragma once


#include "multy-layer-perceptron.h"


MLP::MLP()
{
}

MLP::~MLP()
{
}


/// <summary>
/// foward 
/// </summary>
std::vector<double> MLP::Foward(std::vector<double> input, std::vector<double>* means, std::vector<double>* devs)
{
	std::vector<double> layerInput  =  input;

	for (auto& layer : _layers) {
		std::vector<double> nextLayerInput  =  layer.CalculateLayerOutputs(layerInput, means, devs);
		layerInput  =  nextLayerInput;
	}

	size_t lastLayerIndex  =  _layers.size() - 1;
	std::vector<double> lastLayerOutput  =  _layers[lastLayerIndex].Get<Layer::Attribute::LAYER_OUTPUTS>();

	return std::vector<double>( lastLayerOutput.begin()+1, lastLayerOutput.end() );          // return the predicted output
}



/// <summary>
/// backpropagation
/// </summary>
std::vector<double> MLP::Backward(std::vector<double> correctValues, std::vector<double> predictedValues, std::vector<double> input, size_t* batchSize, bool isBatchNorm)
{
	size_t layerIndex  =  _layers.size() - 1;       // começando de tras para frente


	//--- update last layer
	std::vector<double> inputForLastLayer  =  _layers[layerIndex].Get<Layer::Attribute::RECEIVED_INPUT>();
	if(batchSize != nullptr) { inputForLastLayer = _layers[layerIndex-1].MeanAccumulatedOutput(*batchSize);}
	_layers[layerIndex--].UpdateLastLayerNeurons(correctValues, predictedValues, inputForLastLayer, batchSize, isBatchNorm);


	// update hidden layers
	for (layerIndex; layerIndex > 0; layerIndex--) {
		std::vector<double> gradients  =  _layers[layerIndex+1].Gradients();
		std::vector<double> inputForCurrentLayer  =  _layers[layerIndex].Get<Layer::Attribute::RECEIVED_INPUT>();
		if(batchSize != nullptr) { inputForCurrentLayer = _layers[layerIndex-1].MeanAccumulatedOutput(*batchSize); }

		_layers[layerIndex].UpdateHiddenLayerNeurons(gradients, inputForCurrentLayer, batchSize, isBatchNorm);
	}


	// update the fist hidden layer
	std::vector<double> gradients  =  _layers[layerIndex+1].Gradients();
	if (batchSize == nullptr || isBatchNorm == true) {
		input  =  _layers[0].Get<Layer::Attribute::RECEIVED_INPUT>();
	}
	_layers[0].UpdateHiddenLayerNeurons(gradients, input, batchSize, isBatchNorm);

	return  _layers[0].Gradients();
}



/// <summary>
/// online training
/// </summary>
void MLP::Training(std::vector<TrainigData> trainigSet, std::function<void(void)> callback)
{
	bool keepGoing  =  true;
	size_t epoch  =  0;
	size_t trainingSetSize  =  trainigSet.size();

	while (keepGoing) {

		for (size_t i = 0; i < trainingSetSize; i++) {
			std::vector<double> input  =  trainigSet[i].INPUT;
			std::vector<double> label  =  trainigSet[i].LABEL;

			input.insert(input.begin(), 1.0);


			std::vector<double> predictedOutput  =  Foward( input );
			Backward(label, predictedOutput, input);
		}

		callback();

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(trainigSet.begin(), trainigSet.end(), g);

		ChangeLearningRate(epoch, 0.0);

		epoch++;
		if (epoch > _maxEpochs) {  keepGoing = false;  }
	}

}


void MLP::Training(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback)
{
	std::vector<TrainigData> _trainingSet;

	for (auto data : trainigSet) {
		std::vector<double> label = ParseLabelToVector( data.labelIndex );
		_trainingSet.push_back({ data.input, label });
	}
	
	std::cout << "\n\nstarting training:\n\n";
	Training(_trainingSet, callback);

	BuildJson();
}





/// <summary>
/// Batch training (probabli has some bug)
/// </summary>
void MLP::BatchTraining(std::vector<std::vector<TrainigData>> trainigBatch, std::function<void(void)> callback)
{
	auto& lastLayer  =  _layers[_layers.size()-1];
	size_t neuronsInLastLayer = lastLayer._neurons.size();
	bool keepGoing  =  true;
	size_t epoch  =  0;

	while (keepGoing) {

		for (auto& batch : trainigBatch) {
			size_t batchSize  =  batch.size();
			std::vector<double> batchInput  =  std::vector<double>(batch[0].INPUT.size()+1, 0.0);
			std::vector<double> accumulatedOutput  =  std::vector<double>(batch[0].LABEL.size(), 0.0);
			std::vector<double> accumulatedLabel  =  std::vector<double>(batch[0].LABEL.size(), 0.0);
			std::vector<double> accumulatedGrad  =  std::vector<double>(neuronsInLastLayer, 0.0);

			for (size_t i = 0; i < batch.size(); i++) {
				std::vector<double> input  =  batch[i].INPUT;
				std::vector<double> label  =  batch[i].LABEL;

				input.insert(input.begin(), 1.0);

				std::vector<double> givemOutput  =  Foward( input );
				std::vector<double> gradients  =  lastLayer.CalculateAccumulatedError( label, givemOutput );

				accumulatedOutput  =  Utils::Add(accumulatedOutput, givemOutput);    
				accumulatedLabel  =  Utils::Add(accumulatedLabel, label);
				batchInput  =  Utils::Add( batchInput, input );                   // <--- atualizado dentro do metodo [neuron].UpdateWeights
				//accumulatedGrad  =  Utils::Add(accumulatedGrad, gradients);
			}

			for (auto& out : accumulatedOutput) { out = out / (double)batchSize; }
			for (auto& label : accumulatedLabel) { label = label / (double)batchSize; }
			for (auto& input : batchInput) { input = input / (double)batchSize; }
			//for (auto& grad : accumulatedGrad) { grad = grad / (double)batchSize; }

			//lastLayer.SetGardientForAllNeurons(accumulatedGrad);

			Backward(accumulatedLabel, accumulatedOutput, batchInput, &batchSize);
		}

		callback();

		trainigBatch  =  Utils::ShuffleBatch( trainigBatch, 64 );

		epoch++;
		if (epoch > _maxEpochs) {  keepGoing = false;  }
	}
}


void MLP::BatchTraining(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback)
{

	size_t trainingSetSize  =  trainigSet.size();

	size_t batchIndex  =  0;
	size_t batchSize  =  64;

	std::vector<std::vector<TrainigData>> batchSet  =  Utils::ShuffleBatch(trainigSet, batchSize, ParseLabelToVector);

	BatchTraining(batchSet, callback);

	BuildJson();
}



/// <summary>
/// online training with batch norm
/// </summary>
void MLP::TrainingWithBatchNorm(std::vector<std::vector<TrainigData>> trainigBatch, std::function<void(void)> callback)
{
	auto& lastLayer  =  _layers[_layers.size()-1];
	bool keepGoing  =  true;
	size_t epoch  =  0;

	callback();

	while (keepGoing) {

		for (auto& batch : trainigBatch) {
			std::vector<double> means  =  std::vector(batch[0].INPUT.size(), 0.0);
			std::vector<double> devs   =  std::vector(batch[0].INPUT.size(), 0.0);

			Utils::CalculateMeanVector(batch, &means);
			Utils::CalculateDeviationVector(batch, &means, &devs);

			for (size_t i = 0; i < batch.size(); i++) {
				std::vector<double> input  =  batch[i].INPUT;
				std::vector<double> label  =  batch[i].LABEL;

				input.insert(input.begin(), 1.0);

				std::vector<double> givemOutput  =  Foward( input, &means, &devs );
				Backward(label, givemOutput, input, nullptr, true);
			}
		}

		callback();

		trainigBatch  =  Utils::ShuffleBatch( trainigBatch, 512 );

		ChangeLearningRate(epoch, 0.0);

		epoch++;
		if (epoch > _maxEpochs) {  keepGoing = false;  }
	}
}

void MLP::TrainingWithBatchNorm(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback)
{
	std::vector<double> means  =  std::vector(trainigSet[0].input.size(), 0.0);
	std::vector<double> devs   =  std::vector(trainigSet[0].input.size(), 0.0);

	Utils::CalculateMeanVector(trainigSet, &means);
	Utils::CalculateDeviationVector(trainigSet, &means, &devs);

	for (auto& data : trainigSet) {
		Utils::DataNorm( &data.input, &means, &devs );
	}


	size_t trainingSetSize  =  trainigSet.size();

	size_t batchIndex  =  0;
	size_t batchSize  =  512;

	std::vector<std::vector<TrainigData>> batchSet  =  Utils::ShuffleBatch(trainigSet, batchSize, ParseLabelToVector);

	TrainingWithBatchNorm(batchSet, callback);

	BuildJson();
}




/// <summary>
/// Methods to classify
/// </summary>
std::vector<double> MLP::Classify(std::vector<double> input)
{
	input.insert(input.begin(), 1.0);
	return Foward(input);
}

size_t MLP::Classify(std::vector<double> input, std::function<size_t(std::vector<double>)> ParseOutputToLabel)
{
	input.insert(input.begin(), 1.0);
	std::vector<double> givemOutput  =  Foward( input );
	size_t label  =  ParseOutputToLabel( givemOutput );
	return label;
}

void MLP::Classify(std::vector<std::vector<double>> inputs, std::function<void(std::vector<double>)> CallBack)
{
	for (auto& input : inputs) {
		input.insert(input.begin(), 1.0);
		std::vector<double> givemOutput  =  Foward(input);
		CallBack(givemOutput);
	}
}

void MLP::Classify(std::vector<MLP_DATA> inputSet, std::function<void(std::vector<double>)> CallBack)
{
	for (auto& inputData : inputSet) {
		inputData.input.insert(inputData.input.begin(), 1.0);
		std::vector<double> givemOutput  =  Foward(inputData.input);
		CallBack(givemOutput);
	}
}




/// <summary>
/// auxiliar methods
/// </summary>
void MLP::ChangeLearningRate(size_t epoch, double accuracy)
{
	if (WhenToUpdateLeraningRate(epoch, accuracy)) {
		for (auto& layer : _layers) {
			double newRate  =  HowToUpdateLeraningRate(epoch, accuracy, layer._neuronLerningRate);
			for (auto& neuron : layer._neurons) {
				neuron.Set<Neuron::Attribute::LEARNING_RATE, double>(newRate);
			}
		}
	}
}

Layer& MLP::operator[](size_t layerIndex)
{
	return _layers[layerIndex];
}

Json MLP::ToJson() const
{
	Json mlpJson;

	mlpJson["inputSize"] = _inputSize;

	for (auto& layer : _layers) {
		mlpJson["layers"].push_back(layer.ToJson());
	}

	return mlpJson;

}

void MLP::BuildJson()
{
	if (_outFile != "") {
		Json json = ToJson();

		std::ofstream arquivoSaida(_outFile);


		if (arquivoSaida.is_open()) {
			arquivoSaida << json.dump(4);
			arquivoSaida.close();
		} else {
			std::cerr << "\n\n[ERROR]: could not open file !!! \n\n";
		}
	}
}


