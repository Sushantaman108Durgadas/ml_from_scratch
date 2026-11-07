#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

float randomFloat() {
    return (float)(rand()) / (float)(RAND_MAX);
}

typedef struct {
    int numneurons;
    vector<float> inputvals;
} Layers;

typedef struct {
    int numipneurons;
    int numopneurons;
    int numweights;
    vector<float> weights;
    vector<float> inputvals;
    vector<float> outputvals;
} Operation;

class SOFM {
private:
    Layers InputLayer;
    Layers OutputLayer;
    Operation SOFMOperations;
    int epochs;

    // Find the Best Matching Unit (BMU) using your layered structures
    int findBMU() {
        int bmuIndex = 0;
        float minDist = -1.0;

        int numOp = OutputLayer.numneurons;
        int numIp = InputLayer.numneurons;
        const vector<float>& input = InputLayer.inputvals;

        for (int i = 0; i < numOp; i++) {
            float dist = 0.0;
            for (int j = 0; j < numIp; j++) {
                float diff = input[j] - SOFMOperations.weights[i * numIp + j];
                dist += diff * diff; // Squared Euclidean distance
            }
            dist = sqrt(dist);

            if (minDist == -1.0 || dist < minDist) {
                minDist = dist;
                bmuIndex = i;
            }
        }
        return bmuIndex;
    }

public:
    SOFM(int numipneurons, int numopneurons, int num_epochs) {
        epochs = num_epochs;
        
        // Initialize Input Layer
        InputLayer.numneurons = numipneurons;
        InputLayer.inputvals.resize(numipneurons);
        
        // Initialize Output Layer
        OutputLayer.numneurons = numopneurons;
        OutputLayer.inputvals.resize(numopneurons);
        
        // Initialize Operations struct dimensions & memory
        SOFMOperations.numipneurons = numipneurons;
        SOFMOperations.numopneurons = numopneurons;
        SOFMOperations.numweights = numipneurons * numopneurons;
        SOFMOperations.weights.resize(SOFMOperations.numweights);
        SOFMOperations.inputvals.resize(numipneurons);
        SOFMOperations.outputvals.resize(numopneurons);
        
        // Starting with randomized weights
        srand(time(0));
        for (int i = 0; i < SOFMOperations.numweights; i++) {
            SOFMOperations.weights[i] = randomFloat();
        }
    }

    void train(const vector<vector<float>>& dataset) {
        float initial_lr = 0.1;
        float initial_radius = max(1.0, (double)(OutputLayer.numneurons / 2.0));

        cout << "\n=========================================" << endl;
        cout << "         STARTING SOFM TRAINING          " << endl;
        cout << "=========================================" << endl;

        for (int epoch = 0; epoch < epochs; epoch++) {
            float lr = initial_lr * (1.0 - ((float)epoch / epochs));
            float radius = initial_radius * exp(-(float)epoch / (epochs / log(initial_radius)));

            
            bool printThisEpoch = (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 50 == 0);

            if (printThisEpoch) {
                cout << "\n--- Epoch " << epoch + 1 << "/" << epochs 
                     << " | LR: " << lr << " | Radius: " << radius << " ---" << endl;
            }

            int sampleCount = 0;
            for (const auto& sample : dataset) {
                InputLayer.inputvals = sample;
                int bmu = findBMU();

                if (printThisEpoch) {
                    cout << "  Training Sample [";
                    for (size_t k = 0; k < sample.size(); k++) {
                        cout << sample[k] << (k + 1 < sample.size() ? ", " : "");
                    }
                    cout << "] ---> mapped to BMU Neuron #" << bmu << endl;
                }

                // Update weights of the BMU and its neighborhood
                int numOp = OutputLayer.numneurons;
                int numIp = InputLayer.numneurons;

                for (int i = 0; i < numOp; i++) {
                    float distanceToBMU = abs(i - bmu);

                    if (distanceToBMU <= radius) {
                        float influence = exp(-(distanceToBMU * distanceToBMU) / (2 * radius * radius));

                        for (int j = 0; j < numIp; j++) {
                            int weightIndex = i * numIp + j;
                            SOFMOperations.weights[weightIndex] += lr * influence * (InputLayer.inputvals[j] - SOFMOperations.weights[weightIndex]);
                        }
                    }
                }
                sampleCount++;
            }
        }
        cout << "\n=========================================" << endl;
        cout << "        TRAINING COMPLETED SUCCESSFULLY  " << endl;
        cout << "=========================================" << endl;
    }

    // Helper method to print the final weight vectors of each output neuron
    void printWeights() {
        cout << "\nFinal Learned Weight Vectors for Output Neurons:" << endl;
        int numOp = OutputLayer.numneurons;
        int numIp = InputLayer.numneurons;

        for (int i = 0; i < numOp; i++) {
            cout << "Neuron " << i << " weights: [ ";
            for (int j = 0; j < numIp; j++) {
                cout << SOFMOperations.weights[i * numIp + j] << " ";
            }
            cout << "]" << endl;
        }
    }

    // Prediction / Testing method
    void predict(const vector<float>& sample) {
        InputLayer.inputvals = sample;
        int bmu = findBMU();
        cout << "\n[Prediction Test] Input [";
        for(size_t i=0; i<sample.size(); ++i) {
            cout << sample[i] << (i+1 < sample.size() ? ", " : "");
        }
        cout << "] mapped to Output Neuron (BMU): " << bmu << endl;
    }
};

int main() {
    vector<vector<float>> dataset = {
        {0.1, 0.2},
        {0.9, 0.8},
        {0.2, 0.1},
        {0.8, 0.9}
    };

    // Initialize SOFM: 2 inputs, 4 output neurons, 200 epochs
    SOFM sofmNetwork(2, 4, 200);
    
    // Train the network (prints live mapping status per checkpoint)
    sofmNetwork.train(dataset);

    // Print final weight configuration
    sofmNetwork.printWeights();

    // Test a new sample
    vector<float> testSample = {0.5, 0.8};
    sofmNetwork.predict(testSample);

    return 0;
}
