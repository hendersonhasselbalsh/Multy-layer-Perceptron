#include "layer.h"



/// <summary>
/// Constructor, initialize neurons in this layer
/// </summary>
/// <param name="inputSize">Input vector size</param>
/// <param name="neuronQuantity"></param>
/// <param name="actFun"></param>
/// <param name="neuronLerningRate"></param>
/// <param name="_lostFunction"></param>

Layer::Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun, double neuronLerningRate, ILostFunction* _lostFunction)
	: _inputSize(inputSize), _activationFunction(actFun), _neuronLerningRate(neuronLerningRate)
{
	_neurons = std::vector<Neuron>(neuronQuantity, Neuron(inputSize, actFun, neuronLerningRate, _lostFunction));

	size_t layerOutputSize = _neurons.size() + 1;
	_accumulatedLayerOutouts =  std::vector<double>(layerOutputSize, 0.0);

	_alpha = 1.0;
	_beta  = 0.0;
}




/// <summary>
/// Free pointer
/// </summary>

Layer::~Layer() 
{
	delete _activationFunction;
}




/// <summary>
/// Calculate layer output. Obs: the fist element is always 1.
/// </summary>
/// <param name="inputs">Received input for the layer.</param>
/// <param name="means">Batch mean vector. (Only used in Batch Training)</param>
/// <param name="devs">Batch deviation vector. (Only used in Batch Training)</param>
/// <returns>Vector of output for each neuron activation</returns>

std::vector<double> Layer::CalculateLayerOutputs(std::vector<double> inputs, std::vector<double>* means, std::vector<double>* devs)
{ 
	int outputIndex = 1;

	double* alpha = nullptr;
	double* beta = nullptr;

	if (means!=nullptr && devs!=nullptr) { 
		alpha = &_alpha;
		beta = &_beta;
		Utils::DataNorm(&inputs, means, devs); 
		_receivedInput  =  inputs;
		Utils::ScalateAndShift(&inputs, alpha, beta); 
	}

	_receivedInput  =  inputs;

	for (auto& neuron : _neurons) {
		double output = neuron.CalculateOutput(inputs/*, alpha, beta*/);
		_accumulatedLayerOutouts[outputIndex] += output;							
		_outputs[outputIndex++]  =  output;
	}

	return _outputs;
}




/// <summary>
/// Update all neurons weight. Only used if it's the last layer
/// </summary>
/// <param name="correctValues">Espected Values</param>
/// <param name="predictedValues">Givem Values</param>
/// <param name="inputs">Receives input</param>
/// <param name="batchSize">Param for batch Training</param>
/// <param name="isBatchNorm">Param for Batch Norm</param>
/// <returns></returns>

std::vector<double> Layer::UpdateLastLayerNeurons(std::vector<double> correctValues, std::vector<double> predictedValues, std::vector<double> inputs, size_t* batchSize, bool isBatchNorm)
{
	assert(correctValues.size() == _neurons.size() && predictedValues.size() == _neurons.size());

	if (isBatchNorm) { UpdateBatchNormParms(); }

	int index = 0;
	size_t numberOfNeurons  =  _neurons.size();
	std::vector<double> errors  =  std::vector<double>( numberOfNeurons, 1.0 );

	for (auto& neuron : _neurons) {
		double correctValue  =  correctValues[index];
		double predictedValue = predictedValues[index];
		errors[index++]  =  neuron.CalculateError(correctValue,predictedValue, batchSize);  // [DISCOMENT THIS]
		neuron.UpdateWeights( inputs );
	}

	_accumulatedLayerOutouts = std::vector<double>( (size_t)(_neurons.size()+1), 0.0 );
	_accumulatedLayerOutouts[0] = 1.0;

	return errors;
}




/// <summary>
/// 
/// </summary>
/// <param name="nextLayerGradients">next layer gradient with respect with input</param>
/// <param name="inputs">Receives input</param>
/// <param name="batchSize">Param for batch Training</param>
/// <param name="isBatchNorm">Param for Batch Norm</param>

void Layer::UpdateHiddenLayerNeurons(std::vector<double> nextLayerGradients, std::vector<double> inputs, size_t* batchSize, bool isBatchNorm)
{
	assert(_neurons.size() == nextLayerGradients.size());

	if (isBatchNorm) { UpdateBatchNormParms(); }

	int index = 0;

	for (auto& neuron : _neurons) {
		double gradient = neuron.CalculateGradient(nextLayerGradients[index++], batchSize);         // calcula  (dL/dU) = (dO/dU) * (dL/dO)
		neuron.UpdateWeights( inputs );													            // calcula  (dL/dW) = (dU/dW) * (dL/dU) = (dU/dW) * (dO/dU) * (dL/dO) 
	}

	_accumulatedLayerOutouts = std::vector<double>( (size_t)(_neurons.size()+1), 0.0 );
	_accumulatedLayerOutouts[0] = 1.0;
}




/// <summary>
/// Gradient with respect with input at a certain index.
/// </summary>
/// <param name="index">Input element index.</param>
/// <returns>Loss with respect to that especific index</returns>

double Layer::GradientAtIndex(int index)
{
	double lostGradWithRespectToInput  =  0.0;
	for (auto& neuron : _neurons) {
		lostGradWithRespectToInput  +=  neuron.Gradient(index);
	}
	return lostGradWithRespectToInput;
}




/// <summary>
/// Vector of all partial with respect to input.
/// </summary>
/// <returns></returns>

std::vector<double> Layer::Gradients()
{
	std::vector<double> gradients  =  std::vector<double>( (size_t)_inputSize, 0.0 );

	for (int i = 0; i < _inputSize; i++) {
		gradients[i]  =  GradientAtIndex(i);
	}

	return gradients; 
}

