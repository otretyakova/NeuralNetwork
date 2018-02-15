#include "ReaderFile.h"
#include "NeuralNetwork.h"
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {

	char* trainImagesPath = "data/train-images.idx3-ubyte";
	char* trainLabelsPath = "data/train-labels.idx1-ubyte";
	char* testImagesPath = "data/t10k-images.idx3-ubyte";
	char* testLabelsPath = "data/t10k-labels.idx1-ubyte";

	int numberEpochs = 20;
	double crossError = 0.005;
	double learnRate = 0.01;
	int numberHiddenNeurons = 200;

	switch (argc) {
	case 6:
		numberEpochs = atoi(argv[5]);
		break;
	case 7:
		numberEpochs = atoi(argv[5]);
		crossError = atof(argv[6]);
		break;
	case 8:
		numberEpochs = atoi(argv[5]);
		crossError = atof(argv[6]);
		learnRate = atof(argv[7]);
		break;
	case 9:
		numberEpochs = atoi(argv[5]);
		crossError = atof(argv[6]);
		learnRate = atof(argv[7]);
		numberHiddenNeurons = atoi(argv[8]);
		break;
	}

	int input = 28 * 28;
	int output = 10;
	int sizeTrainData = 60000;
	int sizeTestData = 10000;

	double** trainData = new double*[sizeTrainData];
	for (int i = 0; i < sizeTrainData; i++)
		trainData[i] = new double[input];
	double* trainLabels = new double[sizeTrainData];

	ReadData(trainImagesPath, trainData);
	ReadLabels(trainLabelsPath, trainLabels);

	double** testData = new double*[sizeTestData];
	for (int i = 0; i < sizeTestData; i++)
		testData[i] = new double[input];
	double* testLabels = new double[sizeTestData];

	ReadData(testImagesPath, testData);
	ReadLabels(testLabelsPath, testLabels);

	cout << "Create NeuralNetwork\n";
	NeuralNetwork NN = NeuralNetwork(input, output, learnRate, numberHiddenNeurons);

	cout << "Train: \n";
	NN.Train(trainData, trainLabels, sizeTrainData, numberEpochs, crossError);
	cout << "\n";
	cout << "Test: \n";
	NN.Test(testData, testLabels, sizeTestData, numberEpochs, crossError);

	system("PAUSE");

	for (int i = 0; i < sizeTrainData; i++)
		delete[] trainData[i];
	delete[] trainData;

	for (int i = 0; i < sizeTestData; i++)
		delete[] testData[i];
	delete[] testData;

	delete[] trainLabels;
	delete[] testLabels;

	return 0;
}
