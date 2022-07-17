### Neural network in C++ from scratch


This is my first project in machine learning, a neural network for recognition of handwritten digits and characters.
The code was written without special libraries, and Python was used only to generate the training data from png files. Source of training and testing samples: https://github.com/sueiras/handwritting_characters_database
The network is organized into a binary tree structure, where a each node is a subnetwork making a binary choice. For example, the root subnetwork answers the question "does the digit belong to subset (0,1,2,3,4) or (5,6,7,8,9)", then one of it's children can deal with (0,1) vs (2,3,4) and so on, until a single digit remains. Within a node, the subnetwork is optimized via the usual backpropagation with gradient descent with a constant learning rate.
To run the default version, compile and execute `tree_of_networks.cpp`. 
Current results, each achieved within one hour of training, are given in the table. 
The code and the results are (hopefully) being improved, probably with the use of longer training times. 





| Type of data          | Training set | Test set    | Input layers | Hidden l.  | Output l. | Max iterations per node | Training accuracy | Test accuracy |  
| ----------------------| -------------| ----------- | -------------| ---------- | --------- | ----------------------- | ------------------| ------------- |
| 26 lowercase letters  | ~18k         | ~6k         | 1x175        | 1x100      | 1x1       | 30                      | 95%               | 63%           |
| 26 lowercase letters  | ~18k         | ~6k         | 1x175        | 1x50       | 1x1       | 30 (dev^2 >= 0.05)      | 82%               | 59%           | 
| 26 lowercase letters  | ~18k         | ~6k         | 1x175        | 1x50       | 1x1       | 30                      | 92%               | 63%           | 
| 26 lowercase letters  | ~18k         | ~6k         | 1x175        | 1x50       | 1x1       | 200                     | 97%               | 65%           | 
| 10 digits             | ~7k          | ~2k         | 1x175        | 1x50       | 1x1       | 30                      | 98%               | 80%           | 
| 10 digits             | ~7k          | ~2k         | 1x175        | 1x50, 1x10 | 1x1       | 30                      | 99%               | 80%           | 
| 10 digits             | ~7k          | ~2k         | 1x175        | 100        | 1x1       | 50                      | 99%               | 81%           | 
| 10 digits             | ~7k          | ~2k         | 1x175        | 100        | 1x1       | 100                     | 99%               | 82%           | 
