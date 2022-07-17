#include <bits/stdc++.h>
using namespace std; 
using lli = long long int;


class Network {
public:
    vector<vector<vector<double>>> w;
    vector<vector<double>> b;

    double randn() {
        double u1 = ((double)(rand()%1001 + 1))/1002.0;
        double u2 = ((double)(rand()%1001 + 1))/1002.0;
        return sqrt(-2.0*log(u1))*cos(2*M_PI*u2); 
    }

    Network() {}

    //read saved etwork from file 
    Network(FILE *save, int one_input_size) {
        int num_of_layers; 
        fscanf(save, "%d", &num_of_layers);
        vector<int> layers(num_of_layers);
        for (int l = 0; l < layers.size(); l++) fscanf(save, "%d", &layers[l]); 

        w = vector<vector<vector<double>>>(layers.size());
        for (int l = 0; l <= w.size()-1; l++) w[l] = vector<vector<double>> (layers[l]);
        for (int l = 1; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                w[l][j] = vector<double>(w[l-1].size(), 1); 
            }
        }
        for (int j = 0; j < w[0].size(); j++) {
            w[0][j] = vector<double>(one_input_size); 
        }

        b = vector<vector<double>>(layers.size());
        for (int l = 0; l <= w.size()-1; l++) b[l] = vector<double> (layers[l]);

        for (int l = 0; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                for (int k = 0; k < (l>0 ? w[l-1].size() : one_input_size); k++) {
                    fscanf(save, "%lf", &w[l][j][k]); 
                }
            }
        }

        for (int l = 0; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                fscanf(save, "%lf", &b[l][j]);
            }
        }
    }

    //create a new network
    Network(vector<int> layers, int one_input_size) {
        srand(time(NULL));
        w = vector<vector<vector<double>>>(layers.size());
        for (int l = 0; l <= w.size()-1; l++) w[l] = vector<vector<double>> (layers[l]);
        for (int l = 1; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                w[l][j] = vector<double>(w[l-1].size(), 1); 
            }
        }
        for (int j = 0; j < w[0].size(); j++) {
            w[0][j] = vector<double>(one_input_size); 
        }

        b = vector<vector<double>>(layers.size());
        for (int l = 0; l <= w.size()-1; l++) b[l] = vector<double> (layers[l]);

        for (int l = 0; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                b[l][j] = randn(); 
                for (int k = 0; k < (l>0 ? w[l-1].size() : one_input_size); k++) {
                    w[l][j][k] = randn(); 
                }
            }
        }

    }

    double sigma(double x) {
        return 1.0/(1.0+exp(-x)); 
    }

   vector<vector<double>> forward(vector<double> &one_input) {
        vector<vector<double>> a(w.size()); 
        for (int l = 0; l < w.size(); l++) {
            a[l] = vector<double>(w[l].size()); 
        }
        
        for (int j = 0; j < w[0].size(); j++) {
            for (int k = 0; k < one_input.size(); k++) {
                a[0][j] += one_input[k]*w[0][j][k];
            }
            a[0][j] += b[0][j];
            a[0][j] = sigma(a[0][j]); 
        }
         
        for (int l = 1; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                for (int k = 0; k < w[l-1].size(); k++) {
                    a[l][j] += a[l-1][k]*w[l][j][k];
                }
                a[l][j] += b[l][j];
                a[l][j] = sigma(a[l][j]); 
            }
        }
        
        return a;

    }

    double deviation(vector<vector<double>> &a, vector<double> &one_expected) {
        double dev = 0;
        for (int j = 0; j < a[w.size()-1].size(); j++) dev += pow(a[w.size()-1][j] - one_expected[j], 2);
        return dev; 
    }

    void backward(vector<double> &one_input, vector<double> &one_expected, double d, double L, double &total_deviation) {

        vector<vector<double>> delta_layer(w.size()); 
        for (int l = 0; l < w.size(); l++) {
            delta_layer[l] = vector<double>(w[l].size()); 
        }
        vector<vector<double>> a = forward(one_input);
        double dev0 = deviation(a, one_expected); 
        total_deviation += dev0;

        for (int j = 0; j < w[w.size()-1].size(); j++) {
            // for a custom deviation (loss) function which is not easy to differentiate manually 
            //  a[w.size()-1][j] += d; 
            //  double dev1 = deviation(a, one_expected);
            //  a[w.size()-1][j] -= d; 
            //  delta_layer[w.size()-1][j] = a[w.size()-1][j]*(1.0 - a[w.size()-1][j])*(dev1 - dev0)/d;

            delta_layer[w.size()-1][j] = a[w.size()-1][j]*(1.0 - a[w.size()-1][j])*(a[w.size()-1][j] - (double)one_expected[j]);
        }
    

        
        for (int l = w.size()-2; l >= 0; l--) {
            for (int j = 0; j < w[l].size(); j++) {
                for (int k = 0; k < w[l+1].size(); k++) {
                    delta_layer[l][j] += w[l+1][k][j]*delta_layer[l+1][k]*(1.0 - a[l][j])*a[l][j]; 
                }
            }
        }
    

        
        for (int l = 0; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                b[l][j] -= L*delta_layer[l][j]; 
                for (int k = 0; k < (l>0 ? w[l-1].size() : one_input.size()); k++) {
                    if (l>0) w[l][j][k] -= L*a[l-1][k]*delta_layer[l][j];
                    else w[l][j][k] -= L*one_input[k]*delta_layer[l][j];
                }
            }
        }
        
    }

    void write_to_file(string filename, int one_input_size) {
        FILE *save = fopen(filename.c_str(), "a");
        fprintf(save, "%d\n", one_input_size); 
        for (int l = 0; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                for (int k = 0; k < (l>0 ? w[l-1].size() : one_input_size); k++) {
                    fprintf(save, "%.8f ", w[l][j][k]); 
                }
                fprintf(save, "\n");
            }
            fprintf(save, "\n");
        }
        fprintf(save, "\n");

        for (int l = 0; l < w.size(); l++) {
            for (int j = 0; j < w[l].size(); j++) {
                fprintf(save, "%.8f ", b[l][j]);
            }
            fprintf(save, "\n");
        }
        fprintf(save, "\n\n\n");
        fclose(save);
    }

    void optimize(vector<vector<double>> &input, vector<vector<double>> &expected, double d, double L, int time, double eps) {
        vector<int> order;
        for (int i = 0; i < input.size(); i++) order.push_back(i);

        for (int t = 0; t <= time; t++) {
            random_shuffle(order.begin(), order.end());
            double total_deviation = 0;
            for (int i : order) backward(input[i], expected[i], d, L, total_deviation);
            total_deviation /= (double)(expected.size()*expected[0].size());
            printf("t = %d ... dev = %.6f\n", t, total_deviation);
            if (total_deviation < eps) break;
        }

        write_to_file("network.txt", input[0].size());
    }

};


// int main(void) {
//     double d = 0.25;
//     double L = 0.1;
//     int time = 100; 
//     double EPS = 0.001; 

     
//     vector<int> layers;
//     int one_input_size = 100;
//     layers.push_back(100); 
//     layers.push_back(1);
//     Network net = Network(layers, one_input_size);
//     //Network net = Network("network.txt");
//     vector<vector<double>> input;
//     vector<vector<double>> expected; 
//     int num_of_points;
//     FILE *in = fopen("train.txt", "r");
//     for (int q = 1; q <= 26; q++) {
//         fscanf(in, "%d", &num_of_points);
//         printf("%d , %d\n", q, num_of_points);
//         while(num_of_points--) {
//             vector<double> one_input(one_input_size);
//             for (int i = 0; i < one_input_size; i++) {
//                 fscanf(in, "%lf", &one_input[i]);
//                 one_input[i] = one_input[i]/255.0; 
//             }
//             vector<double> one_expected(1);
//             int ascii; 
//             fscanf(in, "%d", &ascii);
//             one_expected[0] = (int)ascii-97 >= 12 ? 1 : 0; 
//             input.push_back(one_input);
//             expected.push_back(one_expected);
//         }
//     }
     
//     fclose(in);

//     net.optimize(input, expected, d, L, INT_MAX, 0);

//     double acc = 0; 
//     for (int i = 0; i < input.size(); i++) {
//         vector<vector<double>> a = net.forward(input[i]);
//         int des = -1;
//         int mx = -1; 
//         for (int j = 0; j < a[net.w.size()-1].size(); j++) {
//             if (a[net.w.size()-1][j] > mx) {
//                 des = j;
//                 mx = a[net.w.size()-1][j];
//             }
//         }
//         if (des == expected[i][0]) acc++; 
//     }
//     printf("%.4f\n", acc/(double)input.size()); 

//     return 0;
// }