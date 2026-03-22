#include <iostream>
#include <numeric>
#include <cmath>
#include <fstream>
#include <filesystem>

#include "core/Value.h"
#include "core/Tape.h"
#include "core/Node.h"

#include "nn/layers/Linear.h"
#include "nn/models/MLP.h"
#include "nn/optimizers/SGD.h"
#include "nn/losses/MSE.h"

#include "math/random/Normal.h"
#include "math/Statistics.h"

#include "util/data/Dataset.h"
#include "util/data/Dataloader.h"


void test_values() {
    std::cout << "--- Value Test ---" << std::endl;

    Tape tape;
    Tape* tapeP = &tape;

    // Create leaf nodes via Tape
    Value x(tapeP->create_leaf(2.0f), tapeP);
    Value y(tapeP->create_leaf(4.0f), tapeP);
    Value z(tapeP->create_leaf(3.0f), tapeP);

    // Build computation using Value operators
    Value f = (x.pow(2.0f) + y) * z;
    Value result = f.relu();

    // Loss
    Value expected(tapeP->create_leaf(30.0f), tapeP); // Mock expected result
    Value error = expected - result;
    Value loss = error * error; // mean squared error

    // Backprop
    tapeP->zero_grad();
    tapeP->backward(loss.get_node());

    // Output results
    std::cout << "loss = " << loss.get_data() << "\n";
    std::cout << "dloss/dx = " << x.get_grad() << "\n";
    std::cout << "dloss/dy = " << y.get_grad() << "\n";
}

void test_linear() {
    std::cout << "--- Linear Layer Test ---" << std::endl;

    Tape tape;

    Linear l = Linear(&tape, 3, 4, true);

    // Input
    Value i1 = Value(tape.create_leaf(1.0f), &tape);
    Value i2 = Value(tape.create_leaf(2.0f), &tape);
    Value i3 = Value(tape.create_leaf(3.0f), &tape);
    std::vector<Value> input = { i1, i2, i3 };
        
    // Forward
    std::vector<Value> output = l(input);

    // Log
    std::cout << "Output: ";
    for (Value o : output) {
        std::cout << o.get_data() << " ";
    }
    std::cout << std::endl;
}

void test_MLP() {
    std::cout << "--- MLP Test ---" << std::endl;

    Tape tape;

    std::vector<size_t> sizes = { 2, 16, 16, 1 };
    MLP model(&tape, sizes);

    std::cout << "Model: " << std::endl;
    std::cout << model.description() << std::endl;

    // Input
    Value x1(tape.create_leaf(1.0f), &tape);
    Value x2(tape.create_leaf(2.0f), &tape);
    std::vector<Value> input = { x1, x2 };

    // Forward
    std::vector<Value> output = model(input);

    // Loss (L2)
    Value target(tape.create_leaf(3.5f), &tape);
    Value diff = output[0] - target;
    Value loss = diff * diff;

    // Backprop
    tape.zero_grad();
    tape.backward(loss.get_node());

    // Log
    std::cout << "Output: ";
    for (Value o : output) {
        std::cout << o.get_data() << " ";
    }
    std::cout << std::endl;
    std::cout << "loss = " << loss.get_data() << std::endl;
    std::cout << "dloss/dx1 = " << x1.get_grad() << std::endl;
    std::cout << "dloss/dx2 = " << x2.get_grad() << std::endl;
}

void test_SGD() {
    std::cout << "--- SGD Test ---" << std::endl;

    Tape tape;

    std::vector<size_t> sizes = { 2, 16, 16, 1 };
    MLP model(&tape, sizes);
    SGD optimizer = SGD(model.parameters(), 0.0005f);

    std::cout << "Model: " << std::endl;
    std::cout << model.description() << std::endl;
    
    // Debug: print initial parameter count
    std::vector<Value> initial_params = model.parameters();
    std::cout << "Total parameters: " << initial_params.size() << std::endl;
    
    // Input
    Value x1(tape.create_leaf(1.0f), &tape);
    Value x2(tape.create_leaf(2.0f), &tape);
    std::vector<Value> input = { x1, x2 };

    // Target
    Value target(tape.create_leaf(3.5f), &tape);

    // Loss
    MSE loss = MSE(&tape);

    // Loop
    for (size_t i = 0; i < 30; i++) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;
        
        // Forward
        std::vector<Value> output = model(input);

        // Loss (MSE)
        Value lossValue = loss(output[0], target);

        std::cout << "Output: " << output[0].get_data() << std::endl;
        std::cout << "Loss: " << lossValue.get_data() << std::endl;

        // Backprop
        tape.zero_grad();
        tape.backward(lossValue.get_node());

        // Update
        optimizer.step();

        // Clear only computation graph (keeps inputs/targets/parameters valid)
        tape.clear_computation_graph();
    }
}

void test_NormalSample() {
    std::cout << "--- Normal Sample Test ---" << std::endl;

    Normal normalSampler;
    std::vector<float> samples =  normalSampler.sample(0.0f, 1.0f, 500000);

    float mean = math::Statistics::mean(samples);
    float stdDev = math::Statistics::std(samples);

    // Output results
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard Deviation: " << stdDev << std::endl;
}

void test_Dataset() {
    std::cout << "--- Dataset Test ---" << std::endl;

    std::vector<std::vector<float>> X = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f} };
    std::vector<float> y = { 3.5f, 10.0f, 18.0f };
    Dataset dataset(X, y);
    std::cout << "Dataset size: " << dataset.size() << std::endl;
}

void test_DatasetFromCSV() {
    std::cout << "--- Dataset from CSV Test ---" << std::endl;

    try {
        std::filesystem::path dataPath = std::string(PROJECT_ROOT);
        dataPath /= "data";
        dataPath /= "california_housing.csv";

        std::cout << "Loading from: " << dataPath << std::endl;
        Dataset ds = Dataset::from_csv(dataPath.string(), true, ',');

        std::cout << "Dataset size: " << ds.size() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
    }
}

void test_Dataloader() {
    std::cout << "--- Dataloader Test ---" << std::endl;

    try {
        Tape tape;
        std::filesystem::path dataPath = std::string(PROJECT_ROOT);
        dataPath /= "data";
        dataPath /= "california_housing.csv";
        Dataset ds = Dataset::from_csv(dataPath.string(), true, ',');
        Dataloader dl = Dataloader(ds, &tape, true);

        std::vector<Sample> samples = dl.get_epoch_samples();
        size_t num_samples = 5;
        for (size_t i = 0; i < num_samples; i++) {
            Sample s = samples.at(i);
            std::cout << "Sample " << i + 1 << ": X( ";
            for (auto feature : s.X) {
                std::cout << feature.get_data() << " ";
            }
            std::cout << "); y( " << s.y.get_data() << " )" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
    }
}

int main() {
    test_values();
    test_linear();
    test_MLP();
    test_SGD();
    test_NormalSample();
    test_Dataset();
    test_DatasetFromCSV();
    test_Dataloader();

    return 0;
}