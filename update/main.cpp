#include "classifier.cpp"

int main(void) {
    auto sigma = [](double x) -> double {
        return 1.0/(1.0 + exp(-x));
    };
    auto sigmaDer = [sigma](double x) -> double {
        return sigma(x)*(1.0 - sigma(x)); 
    };
    Classifier cl("config.txt", "train.txt", "save.txt", sigma, sigmaDer);
    cl.work("test.txt", "result.txt");
    return 0;
}