#pragma once
#include <bits/stdc++.h>

class NN {
protected:
    std::vector<std::vector<std::vector<double>>> w;
    std::vector<std::vector<double>> b;
    std::function<double(double)> loss;
    int inputSize;
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> forward(const std::vector<double>&); 
public:
    NN(int, std::function<double(double)>);
};

NN::NN(int inputSize, std::function<double(double)> loss) :  inputSize(inputSize), loss(loss) {}

class NNd : public NN {
protected:
    std::function<double(double)> lossDer;
public:
    NNd(int, const std::vector<int>&, std::function<double(double)>, std::function<double(double)>);
    std::vector<double> train(double, const std::vector<double>&, const std::vector<double>&);
    std::vector<double> work(const std::vector<double>&); 
};

NNd::NNd(int inputSize, const std::vector<int>& layersSizes, std::function<double(double)> loss, std::function<double(double)> lossDer) : NN(inputSize, loss), lossDer(lossDer) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    int numOfLayers = layersSizes.size(); 
    w.resize(numOfLayers);
    b.resize(numOfLayers); 
    for (int i = 0; i < numOfLayers; i++) {
        b[i].resize(layersSizes[i]);
        w[i].resize(layersSizes[i]);
        for (int j = 0; j < layersSizes[i]; j++) {
            b[i][j] = dis(gen);
            int sz = (i == 0 ? inputSize : layersSizes[i-1]); 
            w[i][j].resize(sz);
            for (int k = 0; k < sz; k++) {
                w[i][j][k] = dis(gen);
            }
        }
    }
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> NN::forward(const std::vector<double>& input) {
    int numOfLayers = w.size(); 
    std::vector<std::vector<double>> u(numOfLayers); 
    for (int i = 0; i < numOfLayers; i++) {
        u[i] = std::vector<double>(w[i].size()); 
    }
    auto delta = u; 
    for (int i = 0; i < numOfLayers; i++) {
        for (int j = 0; j < w[i].size(); j++) {
            if (i == 0) {
                for (int k = 0; k < input.size(); k++) {
                    u[i][j] += input[k]*w[i][j][k];
                }
            } else {
                for (int k = 0; k < w[i-1].size(); k++) {
                    u[i][j] += loss(u[i-1][k])*w[i][j][k];
                }
            }
            u[i][j] += b[i][j];
        }
    }
    return {u, delta};
}

std::vector<double> NNd::train(double rate, const std::vector<double>& input, const std::vector<double>& expectedOutput) {
    int numOfLayers = w.size();
    auto [u, delta] = forward(input);
    for (int j = 0; j < w[numOfLayers-1].size(); j++) {
        delta[numOfLayers-1][j] = lossDer(u[numOfLayers-1][j])*(loss(u[numOfLayers-1][j]) - expectedOutput[j]);
    }
    for (int i = numOfLayers-2; i >= 0; i--) {
        for (int j = 0; j < w[i].size(); j++) {
            for (int k = 0; k < w[i+1].size(); k++) {
                delta[i][j] += w[i+1][k][j]*delta[i+1][k]*lossDer(u[i][j]); 
            }
        }
    }
    for (int i = 0; i < numOfLayers; i++) {
        for (int j = 0; j < w[i].size(); j++) {
            b[i][j] -= rate*delta[i][j]; 
            if (i == 0) {
                for (int k = 0; k < input.size(); k++) {
                    w[i][j][k] -= rate*input[k]*delta[i][j];
                }
            } else {
                for (int k = 0; k < w[i-1].size(); k++) {
                    w[i][j][k] -= rate*loss(u[i-1][k])*delta[i][j];
                }
            }
        }
    }
    std::vector<double> output;
    for (double v : u.back()) {
        output.push_back(loss(v));
    }
    return output;
}

std::vector<double> NNd::work(const std::vector<double>& input) {
    auto u = forward(input).first;
    std::vector<double> output;
    for (double v : u.back()) {
        output.push_back(loss(v));
    }
    return output;
}