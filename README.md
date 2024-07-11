
##                                                                               Model Training with CUDA Streams and SpecTrain
This repository contains experiments comparing the training latency of various models using standard GPU training and CUDA streams with the SpecTrain optimizer. The experiments include training on randomly generated datasets, CIFAR-10, and FashionMNIST.

## Experiments
## 1. Simple Model with Random Data:

A simple neural network with 2 hidden layers.
Training on a dataset with 100 samples and 10 features.
Comparison of training latency on a single GPU and using CUDA Streams with SpecTrain.
## 2. CNN and Fully Connected Network on CIFAR-10:

Training and comparison of CNN and FC models on the CIFAR-10 dataset.
## 3. CNN and Fully Connected Network on FashionMNIST:

Training and comparison of CNN and FC models on the FashionMNIST dataset.
## 4. ResNet and VGG on CIFAR-10:

Training and comparison of ResNet and VGG models on the CIFAR-10 dataset.
## Dependencies
Ensure you have the following dependencies installed:

torch (version 1.9.0 or later)
torchvision (version 0.10.0 or later)
matplotlib (version 3.4.2 or later)
You can install these dependencies using pip:
pip install torch==1.9.0 torchvision==0.10.0 matplotlib==3.4.2
## Files
main.ipynb: The main Jupyter notebook containing the code for all experiments.
spectrain.py: The implementation of the SpecTrain optimizer.
## Running the Experiments
## 1. Clone the repository:
git clone https://github.com/yourusername/your-repository.git
cd your-repository
## 2. Install dependencies:
pip install torch==1.9.0 torchvision==0.10.0 matplotlib==3.4.2
## 3. Run the notebook:
You can run the main.ipynb notebook using Jupyter Notebook or Jupyter Lab. If you don't have Jupyter installed, you can install it using:
pip install jupyter
Then, start the Jupyter Notebook server:
jupyter notebook
Open main.ipynb in your browser and run the cells to execute the experiments.

## Experiment Details
## Simple Model with Random Data
-->  A simple feedforward neural network with two hidden layers.
-->  Dataset: Randomly generated dataset with 100 samples and 10 features.
-->  Objective: Compare training latency using standard GPU training and CUDA Streams with SpecTrain.
## CNN and Fully Connected Network on CIFAR-10
## Models: Convolutional Neural Network (CNN) and Fully Connected Network (FC).
## Dataset: CIFAR-10, containing 60,000 32x32 color images in 10 classes.
-->  Objective: Compare training latency using standard GPU training and CUDA Streams with SpecTrain.
## CNN and Fully Connected Network on FashionMNIST
## Models: Convolutional Neural Network (CNN) and Fully Connected Network (FC).
## Dataset: FashionMNIST, containing 70,000 grayscale images of 10 fashion categories.
--> Objective: Compare training latency using standard GPU training and CUDA Streams with SpecTrain.
## ResNet and VGG on CIFAR-10
## Models: ResNet18 and VGG16.
D## ataset: CIFAR-10, containing 60,000 32x32 color images in 10 classes.
--> Objective: Compare training latency using standard GPU training and CUDA Streams with SpecTrain.
## Results
The notebook will output the training times for each model and training method, allowing you to compare the performance of standard GPU training and CUDA Streams with the SpecTrain optimizer.

