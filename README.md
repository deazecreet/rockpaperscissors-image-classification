# Rock Paper Scissors Image Classification Project

## Overview
This project involves developing an image classification model to recognize hand gestures representing rock, paper, and scissors. The model is built using Python and TensorFlow/Keras, leveraging convolutional neural networks (CNNs) to achieve high accuracy in predicting these gestures. The goal is to create a robust and reliable model for classifying images into three categories: rock, paper, and scissors.

## Dataset
The dataset consists of 2,188 images of hand gestures categorized into three classes: rock, paper, and scissors. The images were preprocessed to ensure consistent dimensions and normalization. Data augmentation techniques were employed to enhance model training robustness and improve generalization.

## Preprocessing
Data preprocessing steps include:
- Extracting the dataset and organizing it into respective folders.
- Normalizing images to a consistent size of 224x224 pixels.
- Implementing data augmentation techniques such as horizontal and vertical flipping, rotation, zooming, and shifting to increase the diversity of the training data.

## Model Building and Training
The core of this project is building and training a convolutional neural network (CNN) model with the following architecture:
- Three convolutional layers with increasing filter sizes (32, 64, 128) and ReLU activation functions.
- Max-pooling layers following each convolutional layer to reduce spatial dimensions.
- A flatten layer to convert the 2D feature maps into a 1D feature vector.
- Two dense layers, with the final layer using the softmax activation function to output probabilities for the three classes.

The model training process includes:
- Using the Adam optimizer and categorical cross-entropy loss function.
- Implementing early stopping and learning rate reduction callbacks to prevent overfitting and optimize model performance.
- Training the model for 50 epochs with a batch size of 32.

## Results
The model achieved the following performance metrics:
- **Training Accuracy:** 98%
- **Validation Accuracy:** 99%
- **Precision, Recall, and F1-score:** Over 98% for all classes

These results demonstrate the model's high accuracy and reliability in classifying rock, paper, and scissors gestures.

## Model Evaluation
Comprehensive model evaluation was conducted, including:
- Plotting training and validation accuracy and loss over epochs.
- Generating a classification report to compute precision, recall, and F1-score for each class.
- Visualizing the performance metrics to understand the model's behavior and effectiveness.

## Acknowledgments
Credit is given to the creators of the Rock Paper Scissors dataset and any other resources that significantly aided the project.
