#pragma once
#include <bits/stdc++.h>
#include "neural_network.cpp"

class Classifier {
private:
    class Tree : public NNd {
    public:
        std::shared_ptr<Tree> left, right; 
        std::vector<int> size() {
            return {int(w.size()), int(w[0].size()), int(w[0][0].size())};
        }
        Tree(int inputSize, const std::vector<int>& layersSizes, std::function<double(double)> loss, std::function<double(double)> lossDer) : NNd(inputSize, layersSizes, loss, lossDer) {}
    };
    std::shared_ptr<Tree> root;
    int inputSize, numOfItems;
    void assignLayers(std::shared_ptr<Tree> v, int d,
                    const std::vector<int>& layersSizes, 
                    std::function<double(double)> loss, 
                    std::function<double(double)> lossDer) {
        if (d <= 0) {
            return;
        }
        v->left = std::make_shared<Tree>(inputSize, layersSizes, loss, lossDer); 
        v->right = std::make_shared<Tree>(inputSize, layersSizes, loss, lossDer); 
        assignLayers(v->left, d-1, layersSizes, loss, lossDer);
        assignLayers(v->right, d-1, layersSizes, loss, lossDer);
    }
public:
    void work(const std::string& testFileName, const std::string& resultFileName) {
        std::ifstream testFile(testFileName);
        std::ofstream resultFile(resultFileName);
        double correctCnt = 0;
        double totalCnt = 0;
        std::cout << "Working...\n";
        while(!testFile.eof()) {
            totalCnt++;
            std::vector<double> input(inputSize);
            for (int j = 0; j < inputSize; j++) {
                testFile >> input[j];
            }
            int expectedIdx;
            testFile >> expectedIdx;
            int l = 0;
            int r = numOfItems-1;
            auto v = root;
            while (l < r) {
                int m = (l+r)/2;
                auto output = v->work(input);
                if (output[0] < 0.5) {
                    r = m;
                    v = v->left;
                } else {
                    l = m+1;
                    v = v->right;
                }
            }
            if (l == expectedIdx) {
                correctCnt++;
            }
            resultFile << l << "\n";
        }
        std::cout << "Percent of correct answers: " << 100.0*correctCnt/totalCnt << "\n";
        testFile.close();
        resultFile.close();
    }
    Classifier( const std::string& configFileName, 
                const std::string& trainingFileName, 
                const std::string& saveFileName,
                std::function<double(double)> loss,
                std::function<double(double)> lossDer) {
        
        std::ifstream configFile(configFileName);
        double rate;
        int numOfLayers, numOfEpochs;
        std::vector<int> layersSizes; 
        configFile >> numOfEpochs;
        configFile >> numOfItems;
        configFile >> rate;
        configFile >> inputSize;
        configFile >> numOfLayers;
        layersSizes.resize(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            configFile >> layersSizes[i];
        }
        configFile.close();
        root = std::make_shared<Tree>(inputSize, layersSizes, loss, lossDer);
        assignLayers(root, int(log2(inputSize))+2, layersSizes, loss, lossDer);
        for (int epoch = 1; epoch <= numOfEpochs; epoch++) {
            std::cout << "Epoch " << epoch << "\n";
            std::ifstream trainingFile(trainingFileName);
            while (!trainingFile.eof()) {
                std::vector<double> input(inputSize);
                for (int j = 0; j < inputSize; j++) {
                    trainingFile >> input[j];
                }
                int itemIdx;
                trainingFile >> itemIdx;
                int l = 0;
                int r = numOfItems-1;
                auto v = root; 
                while (l < r) {
                    int m = (l+r)/2;
                    bool isLeft = (itemIdx <= m);
                    auto output = v->train(rate, input, std::vector<double>(1, isLeft ? 0 : 1));
                    if (isLeft) {
                        r = m;
                        v = v->left;
                    } else {
                        l = m+1;
                        v = v->right;
                    }
                }
            }
            trainingFile.close();
            work("test.txt", "result.txt");
        }
    } 
};
 