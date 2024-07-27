#pragma once

#include "basic-includes.h"
#include "utils.h"
#include "layer.h"

#define INPUT first
#define LABEL second
using TrainigData = std::pair<std::vector<double>, std::vector<double>>;


class MlpBuilder;



class MLP {

	private:
	//--- atributos importantes do mlp
		std::vector<Layer> _layers;
		ILossFunction* _lostFunction;

		std::function<std::vector<double>(size_t)> ParseLabelToVector;
		std::function<bool(size_t, double)> WhenToUpdateLeraningRate;             //  bool f(size_t epoch, double accuracy);
		std::function<double(size_t, double, double)> HowToUpdateLeraningRate;    //  double f(size_t epoch, double accuracy, double currentLearningRate)
		std::string _outFile;

		size_t _inputSize;
		size_t _maxEpochs;
		double _acceptableAccuracy;
		std::vector<double> _accumulatedGradients;


	//--- private constructor, used by builder
		MLP();


	//--- foward fase and backpropagation
		std::vector<double> Foward(std::vector<double> input, std::vector<double>* means = nullptr, std::vector<double>* devs = nullptr);
		std::vector<double> Backward(std::vector<double> correctValues, std::vector<double> predictedValues, std::vector<double> input, size_t* batchSize = nullptr, bool isBatchNorm = false);      // In batch training instead of the correctValue (labelVector), you should pass the accumulated gradient

		void BuildJson();
		Json ToJson() const;

		void ChangeLearningRate(size_t epoch, double accuracy);


	public:
		~MLP();

	//--- training loops
		void Training(std::vector<TrainigData> trainigSet, std::function<void(void)> callback = [](){} );
		void Training(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback = [](){} );

		void BatchTraining(std::vector<std::vector<TrainigData>> trainigBatch, std::function<void(void)> callback = [](){});
		void BatchTraining(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback = [](){} );

		void TrainingWithBatchNorm(std::vector<std::vector<TrainigData>> trainigBatch, std::function<void(void)> callback = [](){});
		void TrainingWithBatchNorm(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback = [](){} );

	//--- classify methods
		std::vector<double> Classify(std::vector<double> input);
		size_t Classify(std::vector<double> input, std::function<size_t(std::vector<double>)> ParseOutputToLabel);
		void Classify(std::vector<std::vector<double>> inputSet, std::function<void(std::vector<double>)> CallBack);
		void Classify(std::vector<MLP_DATA> inputSet, std::function<void(std::vector<double>)> CallBack);


		Layer& operator[](size_t layerIndex);


	friend class MlpBuilder;
};


#include "mlp-builder.h"

