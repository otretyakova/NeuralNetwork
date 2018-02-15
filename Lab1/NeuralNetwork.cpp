#include "NeuralNetwork.h"

double random(double min, double max)
{
	return (double)(rand()*(max - min) / ((double)RAND_MAX + 0.1) + min);
};

NeuralNetwork::NeuralNetwork(int input_n, int output_n, double learningRate, int numberOfHiddenNeurons)
{
	this->input_n = input_n;
	this->output_n = output_n;
	this->learningRate = learningRate;
	this->numberOfHiddenNeurons = numberOfHiddenNeurons;

	input = new double[input_n];
	output = new double[output_n];
	hide_output = new double[numberOfHiddenNeurons];

	deltaWeightOnHide = new double[numberOfHiddenNeurons];
	deltaWeightOnOutput = new double[output_n];
	for (int j = 0; j < output_n; j++)
		deltaWeightOnOutput[j] = random(-1.0, 1.0);

	for (int j = 0; j < numberOfHiddenNeurons; j++)
		deltaWeightOnHide[j] = random(-1.0, 1.0);

	sumHide = new double[numberOfHiddenNeurons];
	sumOutput = new double[output_n];
	GradH = new double[numberOfHiddenNeurons];
	GradO = new double[output_n];

	z_exp = new double[output_n];
	soft_max = new double[output_n];

	hideWeights = memory_weights(input_n, numberOfHiddenNeurons);
	outputWeights = memory_weights(numberOfHiddenNeurons, output_n);

	for (int i = 0; i < input_n; i++)
		for (int j = 0; j < numberOfHiddenNeurons; j++)
			hideWeights[i][j] = random(-1.0, 1.0);

	for (int i = 0; i < numberOfHiddenNeurons; i++)
		for (int j = 0; j < output_n; j++)
			outputWeights[i][j] = random(-1.0, 1.0);

	nul_mas(numberOfHiddenNeurons, sumHide);
	nul_mas(output_n, sumOutput);
	nul_mas(numberOfHiddenNeurons, GradH);
	nul_mas(output_n, GradO);
};

void NeuralNetwork::Train(double** data, double* labels, int data_size, int numberOfEpochs, double crossError)
{
	double* T = new double[output_n];
	int epoch_count = 0;
	while (epoch_count < numberOfEpochs)
	{
		double ok = 0.0;
		Mix(data, labels, data_size);
		cout << "Number of ep = " << epoch_count << "\n";

		cout << "Calculating output and changing weights...\n";
		for (int i = 0; i < data_size; i++)
		{
			for (int j = 0; j < input_n; j++)
				input[j] = data[i][j];

			for (int j = 0; j < output_n; j++)
			{
				T[j] = 0.0;
				if (j == labels[i])
					T[j] = 1.0;
			}

			output = check_output(check_hide_output(input));
			if (T[find_max(output, output_n)] == 1.0)
				ok++;

			Check_grad(T);

			Change_Weights(GradO, GradH);
			Change_Delta(GradO, GradH);
		}
		cout << " Computing cross-entrophy" << endl;
		double cross = 0.0;
		cross = CrossEntropy(data, labels, data_size);
		cout << "Cross-entrophy: " << cross << " ";
		cout << endl;
		cout << "Number of right answers = " << ok << endl;
		double error_calc = 0.0;
		error_calc = ok / data_size;
		cout << "Part of wrong answers = " << (1 - error_calc) << endl;
		epoch_count++;

		if ((cross <= crossError) || (1 - error_calc <= crossError))
			break;
	}
};
void NeuralNetwork::Test(double** data, double* labels, int data_size, int numberOfEpochs, double crossError)
{
	double* T = new double[output_n];
	int epoch_count = 0;
			double ok = 0.0;
		Mix(data, labels, data_size);
		cout << "Calculating output and changing weights\n";
		for (int i = 0; i < data_size; i++)
		{
			for (int j = 0; j < input_n; j++)
				input[j] = data[i][j];

			for (int j = 0; j < output_n; j++)
			{
				T[j] = 0.0;
				if (j == labels[i])
					T[j] = 1.0;
			}

			output = check_output(check_hide_output(input));
			if (T[find_max(output, output_n)] == 1.0)
				ok++;

		}
		cout << " Computing cross-entrophy" << endl;
		double cross = 0.0;
		cross = CrossEntropy(data, labels, data_size);
		cout << "Cross-entrophy: " << cross << " ";
		cout << endl;
		cout << "Number of right answers = " << ok << endl;
		double error_calc = 0.0;
		error_calc = ok / data_size;
		cout << "Part of wrong answers = " << (1 - error_calc) << endl;
};

int NeuralNetwork::find_max(double *mas, int size)
{
	double max = 0.0;
	int index = 0;
	for (int i = 0; i < size; i++)
		if (max < mas[i])
		{
			max = mas[i];
			index = i;
		}

	return index;
};
double* NeuralNetwork::check_hide_output(double * input)
{
	nul_mas(numberOfHiddenNeurons, sumHide);
	for (int i = 0; i< numberOfHiddenNeurons; i++)
		for (int j = 0; j < input_n; j++)
			sumHide[i] += input[j] * hideWeights[j][i];

	for (int i = 0; i < numberOfHiddenNeurons; i++)
		sumHide[i] += deltaWeightOnHide[i];

	for (int i = 0; i < numberOfHiddenNeurons; i++)
		hide_output[i] = BinSigmFun(sumHide[i]);

	return hide_output;
};

double* NeuralNetwork::check_output(double *hide_output)
{
	nul_mas(output_n, sumOutput);
	for (int i = 0; i < output_n; i++)
		for (int j = 0; j < numberOfHiddenNeurons; j++)
			sumOutput[i] += hide_output[j] * outputWeights[j][i];

	for (int i = 0; i < output_n; i++)
		sumOutput[i] += deltaWeightOnOutput[i];

	output = softmax(sumOutput);
	return output;
};

double NeuralNetwork::BinSigmFun(double x)
{
	return 1.0 / (1.0 + exp(-x));
};

double** NeuralNetwork::memory_weights(int before, int current)
{
	double ** mas;
	mas = new double *[before];
	for (int i = 0; i<before; i++)
		mas[i] = new double[current];

	return mas;
};

void NeuralNetwork::nul_mas(int size, double* mas)
{
	for (int i = 0; i<size; i++)
		mas[i] = 0;
};

double* NeuralNetwork::softmax(double* sumOut)
{
	for (int i = 0; i< output_n; i++)
		z_exp[i] = exp(sumOut[i]);

	double sum = 0;
	for (int i = 0; i< output_n; i++)
		sum += z_exp[i];

	for (int i = 0; i< output_n; i++)
		soft_max[i] = z_exp[i] / sum;

	return soft_max;
};



void NeuralNetwork::check_grad_out(double * t)
{
	for (int i = 0; i < output_n; i++)
		GradO[i] = (t[i] - output[i]);
};

void NeuralNetwork::Check_grad(double * T)
{
	check_grad_out(T);

	double sum = 0.0;
	double der = 0.0;
	for (int i = 0; i < numberOfHiddenNeurons; i++)
	{
		for (int j = 0; j < output_n; j++)
			sum += GradO[j] * outputWeights[i][j];

		der = hide_output[i] * (1 - hide_output[i]);
		GradH[i] = sum*der;
	}
};

void NeuralNetwork::Change_Weights(double * Grad_o, double * Grad_h)
{
	double delta = 0.0;
	for (int i = 0; i < numberOfHiddenNeurons; i++)
		for (int j = 0; j < output_n; j++)
		{
			delta = learningRate*Grad_o[j] * hide_output[i] * 0.9;
			outputWeights[i][j] += delta;
		}

	for (int i = 0; i < input_n; i++)
		for (int j = 0; j < numberOfHiddenNeurons; j++)
		{
			delta = learningRate*Grad_h[j] * input[i];
			hideWeights[i][j] += delta;
		}
};

void NeuralNetwork::Change_Delta(double * Grad_o, double * Grad_h)
{
	double delta = 0.0;
	for (int j = 0; j < output_n; j++)
	{
		delta = learningRate*Grad_o[j] * 0.9;
		deltaWeightOnOutput[j] += delta;
	}

	for (int j = 0; j < numberOfHiddenNeurons; j++)
	{
		delta = learningRate*Grad_h[j];
		deltaWeightOnHide[j] += delta;
	}

};

double NeuralNetwork::CrossEntropy(double** TrainDataSet, double* Labels, int size)
{
	double sum = 0.0;

	double *X = new double[input_n];
	double *Y = new double[output_n];
	double *T = new double[output_n];

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < input_n; j++)
			X[j] = TrainDataSet[i][j];

		for (int j = 0; j < output_n; j++)
		{
			T[j] = 0.0;
			if (j == Labels[i])
				T[j] = 1.0;
		}

		Y = check_output(check_hide_output(X));
		for (int j = 0; j < output_n; j++)
			sum += log(Y[j]) * T[j];
	}
	return (-1)*(sum / size);
};

void NeuralNetwork::Mix(double** Dataset, double* Labels, int size)
{
	for (int i = 0; i<size; i++)
	{
		int nom1 = rand() % size;
		int nom2 = rand() % size;

		swap(Dataset[nom1], Dataset[nom2]);
		swap(Labels[nom1], Labels[nom2]);
	}
}

NeuralNetwork :: ~NeuralNetwork(void)
{
	delete[] input;
	delete[] output;
	delete[] hide_output;
	delete[] deltaWeightOnHide;
	delete[] deltaWeightOnOutput;
	delete[] sumHide;
	delete[] sumOutput;
	delete[] GradH;
	delete[] GradO;
	delete[] z_exp;
	delete[] soft_max;

	for (int i = 0; i < input_n; i++)
		delete[] hideWeights[i];
	delete[] hideWeights;

	for (int i = 0; i < numberOfHiddenNeurons; i++)
		delete[] outputWeights[i];
	delete[] outputWeights;

	delete[] hideWeights;
	delete[] outputWeights;
}