#include <stdio.h> 

#include <conio.h> 

#include <stdlib.h> 

#include <math.h> 

#include <time.h> 

  

#define numInputs 4 

#define numHiddenNodes 4 

#define numOutputs 1 

#define numTrainingSets 18 

#define numTestSets 5 

  

  

double sigmoid(double x) { 

	return 1 / (1 + exp(-x)); 

} 

double dSigmoid(double x) { 

	return exp(-x)/((1 + exp(-x) * (1 + exp(-x)))); 

} 

double init_weight() { 

	srand(time(NULL));

	return ((double)rand()/(double)RAND_MAX); 

} 
void input_2D_data(double input[numTrainingSets][numInputs], size_t rows, size_t cols) {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			scanf("%lf", &input[i][j]);
			input[i][j] /= 1E12;
		}
	}
}
void output_2D_data(double output[numTrainingSets][numOutputs], size_t rows, size_t cols) {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			scanf("%lf", &output[i][j]);
			output[i][j] /= 1E12;
		}
	}
}

void shuffle(int *array, size_t n) { 

	if(n > 1) { 

		size_t i; 

		for(i = 0; i < n - 1; i++) { 

			size_t j = i + rand() / (RAND_MAX / (n-i)); 

			int t = array[j]; 

			array[j] = array[i]; 

			array[i] = t; 

		} 

	} 

} 

  

int main() { 

freopen("input.txt" , "r", stdin);
	const double lr = 0.1f;

  

	double hiddenLayer[numHiddenNodes]; 

	double outputLayer[numOutputs]; 

  

	double hiddenLayerBias[numHiddenNodes]; 

	double outputLayerBias[numOutputs]; 

  

	double hiddenWeights[numInputs][numHiddenNodes]; 

	double outputWeights[numHiddenNodes][numOutputs]; 

	 

	double test_inputs[numTestSets][numInputs] = {{0.116, 0.0969, 0.168, 0.127}, 

												  {0.0969, 0.168, 0.127, 0.104}, 

												  {0.168, 0.127, 0.104, 0.0574}, 

												  {0.127, 0.104, 0.0574, 0.197}, 

												  {0.104, 0.0574, 0.197, 0.09640612} 

												  }; 

	double test_outputs[numTestSets][numOutputs] = {{0.104}, 

													{0.0574}, 

													{0.197}, 

													{0.09640612}, 

													{0.08159862} 

													}; 

  

	double training_inputs[numTrainingSets][numInputs];
	input_2D_data(training_inputs, numTrainingSets, numInputs);
	
	double training_outputs[numTrainingSets][numOutputs];
	output_2D_data(training_outputs, numTrainingSets, numOutputs);

	//Initialize weights and biases
	for(int i = 0; i < numInputs; i++) { 

		for(int j = 0; j < numHiddenNodes; j++) { 

			hiddenWeights[i][j] = init_weight(); 

		} 

	} 

	for(int i = 0; i < numHiddenNodes; i++) { 

		for(int j = 0; j < numOutputs; j++) { 

			outputWeights[i][j] = init_weight(); 

		} 

	} 

	for(int i = 0; i < numHiddenNodes; i++) { 

		hiddenLayerBias[i] = init_weight(); 

	} 

	for(int i = 0; i < numOutputs; i++) { 

		outputLayerBias[i] = init_weight(); 

	} 

 
	int trainingSetOrder[numTrainingSets] = {0, 1, 2, 3, 4, 5, 6, 7, 8,  

							  9, 10, 11, 12, 13, 14, 15, 16, 17}; 

  

	int numberOfEpochs = 10000; 

  

	//Train the neural network 

	for(int epoch = 0; epoch < numberOfEpochs; epoch++) { 

		shuffle(trainingSetOrder, numTrainingSets); 

		for(int x = 0; x < numTrainingSets; x++) { 

			int i = trainingSetOrder[x]; 

			//Forward pass 

			//Compute hidden layer activation 

			for(int j = 0; j < numHiddenNodes; j++) { 
				double activation = hiddenLayerBias[j]; 
				for(int k = 0; k < numInputs; k++) { 
					activation += training_inputs[i][k] * hiddenWeights[k][j]; 
				} 
				hiddenLayer[j] = sigmoid(activation); 
			} 

			//Compute output layer activation 

			for(int j = 0; j < numOutputs; j++) { 
				double activation = outputLayerBias[j]; 
				for(int k = 0; k < numHiddenNodes; k++) { 
					activation += hiddenLayer[k] * outputWeights[k][j]; 
				} 
				outputLayer[j] = sigmoid(activation); 
			} 
			printf("Input: %.8lf         %.8lf         %.8lf          %.8lf         Output: %.8lf        Predicted Output: %.8lf\n",  
				training_inputs[i][0], training_inputs[i][1], training_inputs[i][2], training_inputs[i][3], 
				outputLayer[0], training_outputs[i][0]); 

			//Backprop

			//Compute change in output weights 
  
			double deltaOutput[numOutputs]; 

			for(int j = 0; j < numOutputs; j++){ 
				double error = (training_outputs[i][j] - outputLayer[j]); 
				deltaOutput[j] = error * dSigmoid(outputLayer[j]); 
			} 

			//Compute change in hidden weights 

			double deltaHidden[numHiddenNodes]; 

			for(int j = 0; j < numHiddenNodes; j++){ 
				double error = 0.0f; 
				for(int k = 0; k < numOutputs; k++) { 
					error += deltaOutput[k] * outputWeights[j][k]; 
				} 
				deltaHidden[j] = error * dSigmoid(hiddenLayer[j]); 
			}

			//Apply change in output weights 
			for(int j = 0; j < numOutputs; j++) { 
				outputLayerBias[j] += deltaOutput[j] * lr; 
				for(int k = 0; k < numHiddenNodes; k++) { 
					outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr; 
				} 
			}

			//Apply change in hidden weights 
			for(int j = 0; j < numHiddenNodes;j++) { 
				hiddenLayerBias[j] += deltaHidden[j] * lr; 
				for(int k = 0; k < numInputs; k++) { 
					hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr; 
				} 
			} 
		} 
	} 

	for(int i = 0; i < numTestSets; i++) { 
		//Pre_process hidden layer 
		for(int j = 0; j < numHiddenNodes; j++) { 
			hiddenLayer[j] = hiddenLayerBias[j]; 
			for(int k = 0; k < numInputs; k++) { 
				hiddenLayer[j] += test_inputs[i][k] * hiddenWeights[k][j]; 
				//printf("%lf", hiddenLayer[j]); 
			} 
			hiddenLayer[j] = sigmoid(hiddenLayer[j]); 
		} 
		//Predict 
		for(int j = 0; j < numOutputs; j++) { 
			outputLayer[j] = outputLayerBias[j]; 
			for(int k = 0; k < numHiddenNodes; k++) { 
				outputLayer[j] += hiddenLayer[k] * outputWeights[k][j]; 
			} 
			outputLayer[j] = sigmoid(outputLayer[j]); 
		} 
		printf("Test: \n"); 
		printf("Predict Output: %.8lf     Expected Output: %.8lf     \n", outputLayer[i], test_outputs[i][0]); 

	} 
	//getch(); 
} 
