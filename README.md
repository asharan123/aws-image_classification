# Sattilite Image Classification with TensorFlow on Amazon SageMaker

## Overview
This repository contains the code for training a bird classification model using TensorFlow on Amazon SageMaker. The model is designed to classify images taken by sattelite into different weather.

## LeNet Architecture
The sattelite classification model is built using the LeNet architecture, a classic convolutional neural network (CNN) developed by Yann LeCun and his colleagues. LeNet consists of the following layers:

Convolutional Layer (Conv2D): The first layer applies convolutional filters to the input images to extract features.

Average Pooling Layer (AveragePooling2D): Down-samples the spatial dimensions of the input by taking the average value over a small region.

Flatten Layer: Flattens the output from the previous layer into a one-dimensional vector.

Dense Layers: Fully connected layers that perform classification based on the learned features.

Dense Layer 1: 512 units, ReLU activation.

Dense Layer 2: 512 units, ReLU activation.

Output Layer: 4 units (number of weather), softmax activation.

## Activation Function:
ReLU (Rectified Linear Unit): Used as the activation function in the convolutional and dense layers. ReLU introduces non-linearity to the model by outputting the input for positive values and zero for negative values.
Optimizer:
Adam Optimizer: A popular optimization algorithm that adapts the learning rates of each parameter during training. Adam combines the advantages of two other extensions of stochastic gradient descent, AdaGrad and RMSProp.

## Metrics:
Categorical Crossentropy Loss: The loss function used for multi-class classification problems. It measures the difference between the predicted probabilities and the true labels.
Accuracy: A common metric for classification problems, measuring the proportion of correctly classified samples.

## Project Structure
train.py: The main script for training the bird classification model.
awsipynb.ipynb: Jupyter Notebook containing the step-by-step process and execution of the training script.
data: CSV file containing file paths and labels for the training and validation datasets.

# Instructions
## Setup Environment:

Ensure you have Python 3.10 installed.
Install necessary Python packages by running: pip install -r requirements.txt.
Training:

Run the train.py script to train the bird classification model. Adjust hyperparameters in the script if needed.
Notebook Execution:

Open and run the awsipynb.ipynb Jupyter Notebook for a detailed walkthrough and execution of the training script.
CSV File Update:

If the birds.csv file is not available locally, it will be downloaded from the specified S3 bucket during script execution.

## Amazon SageMaker Integration
The script is designed to be run on Amazon SageMaker. Ensure your SageMaker environment is properly set up.

Update the entry_point parameter in the TensorFlow estimator in the notebook with the path to your train.py script.

Set the correct S3 bucket and path for the s3_csv_path in the script to download the birds.csv file from S3.

## Acknowledgments
Thanks to Kaggle for providing the sattelite dataset.