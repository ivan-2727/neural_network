#include <bits/stdc++.h>
#include "network.cpp"
using namespace std; 
using lli = long long int;

struct Data {
    vector<vector<double>> input;
    vector<vector<double>> expected; 
};
 
class Node {
public:
    vector <int> codes;
    Node* zero;
    Node* one; 
    Network net;
    int one_input_size;
    Node (vector <int> _codes, vector <int> layers, int _one_input_size) {
        codes = _codes;
        one_input_size = _one_input_size;
        net = Network(layers, one_input_size);
        zero = nullptr;
        one = nullptr;
    }
    Data read_data(string train_file) {
        vector<vector<double>> input;
        vector<vector<double>> expected;
        FILE *in = fopen(train_file.c_str(), "r");
        int total_num_of_classes;
        fscanf(in, "%d", &total_num_of_classes); 
        while(total_num_of_classes--) {
            int num_of_points; 
            fscanf(in, "%d", &num_of_points);
            while(num_of_points--) {
                vector<double> one_input(one_input_size);
                for (int i = 0; i < one_input_size; i++) {
                    fscanf(in, "%lf", &one_input[i]);
                    one_input[i] = one_input[i]/255.0; 
                }
                int ascii; fscanf(in, "%d", &ascii);
                vector<double> one_expected(1);
                if (ascii >= codes[0] and ascii < codes[codes.size()/2]) {
                    one_expected[0] = 0;
                    expected.push_back(one_expected);
                    input.push_back(one_input);
                }
                if (ascii >= codes[codes.size()/2] and ascii <= codes[codes.size()-1]) {
                    one_expected[0] = 1; 
                    expected.push_back(one_expected);
                    input.push_back(one_input);
                }
            }
        }
        fclose(in); 
        return {input, expected};
    }

    void optimize(string train_file) {
        if (codes.size()>1) {
            Data data = read_data(train_file);
            net.optimize(data.input, data.expected, 0.1, 0.1, 50, 0);
        }
    }

    int choose_half(vector<double> &one_input) {
        vector<vector<double>> a = net.forward(one_input);
        double res = a[net.w.size()-1][0];
        if (abs(res) < abs(res-1)) return 0;
        return 1; 
    }
};

Node* generate_tree(vector<int> codes, vector <int> layers, int one_input_size) {
    Node* root = new Node(codes, layers, one_input_size); 
    if (codes.size()==1) return root;
    vector <int> left, right; 
    for (int i = 0; i < codes.size()/2; i++) left.push_back(codes[i]);
    for (int i = codes.size()/2; i < codes.size(); i++) right.push_back(codes[i]);
    root->zero = generate_tree(left, layers, one_input_size);
    root->one = generate_tree(right, layers, one_input_size);
    return root; 
}

void optimize_tree(Node* v, string training_file) {
    if (v) {
        for (int x : v->codes) printf("%d ", x);
        printf("\n"); 
        v->optimize(training_file); 
        optimize_tree(v->zero, training_file);
        optimize_tree(v->one, training_file);
    }
}

void read_node(FILE* &save, Node *v) {
    if (v) if (v->codes.size()>1) {

        int one_input_size;
        fscanf(save, "%d", &one_input_size); 

        for (int l = 0; l < v->net.w.size(); l++) {
            for (int j = 0; j < v->net.w[l].size(); j++) {
                for (int k = 0; k < (l>0 ? v->net.w[l-1].size() : one_input_size); k++) {
                    fscanf(save, "%lf", &v->net.w[l][j][k]); 
                }
            }
        }

        for (int l = 0; l < v->net.w.size(); l++) {
            for (int j = 0; j < v->net.w[l].size(); j++) {
                fscanf(save, "%lf", &v->net.b[l][j]);
            }
        }

        read_node(save, v->zero);
        read_node(save, v->one);
    }
}


void read_tree_from_file(Node* v, string saved_tree) {
    FILE *save = fopen(saved_tree.c_str(), "r");
    read_node(save, v);
    fclose(save);
}

int predict(Node* v, vector <double> &one_input) {
    if (v->codes.size() == 1) return v->codes[0];
    int half = v->choose_half(one_input);
    if (half == 0) return predict(v->zero, one_input);
    return predict(v->one, one_input);
}

double test(Node* root, string test_file, int one_input_size) {
    FILE *in = fopen(test_file.c_str(), "r");
    double correct = 0; 
    double total = 0; 
    int total_num_of_classes;
    fscanf(in, "%d", &total_num_of_classes); 
    while(total_num_of_classes--) {
        int num_of_points; 
        fscanf(in, "%d", &num_of_points);
        while(num_of_points--) {
            vector<double> one_input(one_input_size);
            for (int i = 0; i < one_input_size; i++) {
                fscanf(in, "%lf", &one_input[i]);
                one_input[i] = one_input[i]/255.0; 
            }
            int ascii; fscanf(in, "%d", &ascii);
            if (predict(root, one_input) == ascii) correct++; 
            total++; 
        }
    }
    fclose(in); 
    return correct/total; 
}

int main(void) {
    int one_input_size = 15*15; 
    vector <int> codes;
    //for (int i = 97; i <= 122; i++) codes.push_back(i); 
    for (int i = 48; i <= 57; i++) codes.push_back(i); 
    Node* root = generate_tree(codes, {100,1}, one_input_size);
     
    read_tree_from_file(root, "network.txt"); 

    //MAKE SURE TO COMMENT/UNCOMMENT OUT IF NECESSARY
    //FILE *delete_saved = fopen("network.txt", "w"); fclose(delete_saved); 
    
    //optimize_tree(root, "train.txt");

    printf("\nTraining accuracy: %.4f\n", test(root, "train.txt", one_input_size));
    printf("\nTest accuracy: %.4f\n", test(root, "test.txt", one_input_size));
    return 0;
}