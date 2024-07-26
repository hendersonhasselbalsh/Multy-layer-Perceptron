#pragma once

class ILossFunction {
	public:
		virtual double f(double predicted, double correct) = 0;
		virtual double df(double predicted, double correct) = 0;
};




//--- mean absolute error
class MAE : public ILossFunction {
	public:
		virtual double f(double predicted, double correct) override;
		virtual double df(double predicted, double correct) override;
};



//--- mean square error
class MSE : public ILossFunction {
	public:
		virtual double f(double predicted, double correct) override;
		virtual double df(double predicted, double correct) override;
};



//--- Root Mean Square error
class RMSE : public ILossFunction {
	public:
		virtual double f(double predicted, double correct) override;
		virtual double df(double predicted, double correct) override;
};



//--- Cross Entropy
class CrossEntropy : public ILossFunction {
	public:
		virtual double f(double predicted, double correct) override;
		virtual double df(double predicted, double correct) override;
};
