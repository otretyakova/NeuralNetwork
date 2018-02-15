#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(int input_n, int output_n, double learningRate, int numberOfHiddenNeurons);
	void Train(double** TrainDataSet, double* Labels, int data_size, int numberOfEpochs, double crossError);
	void Test(double** TrainDataSet, double* Labels, int data_size, int numberOfEpochs, double crossError);	
	double * check_output(double * hide_output);
	~NeuralNetwork();

private:
	int input_n;
	int output_n;
	double learningRate;
	int numberOfHiddenNeurons;

	double ModCoef;

	double** hideWeights;
	double** outputWeights;

	double* input;
	double* output;
	double* hide_output;
	double* deltaWeightOnHide;
	double* deltaWeightOnOutput;
	double* sumHide;
	double* sumOutput;
	double* GradH;
	double* GradO;

	double * z_exp;
	double * soft_max;

	double** memory_weights(int before, int current);
	void nul_mas(int size, double* mas);
	double* softmax(double* X);
	double* check_hide_output(double * input);
	double BinSigmFun(double x);
	void check_grad_out(double *y);
	void Check_grad(double *y);
	void Change_Weights(double * Grad_o, double * Grad_h);
	void Change_Delta(double * Grad_o, double * Grad_h);
	double CrossEntropy(double** TrainDataSet, double* Labels, int size);
	void Mix(double** Dataset, double* Labels, int size);
	int find_max(double *mas, int size);
};