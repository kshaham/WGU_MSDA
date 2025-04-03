# 💬 Summary of WGU D213 Task 2: Sentiment Analysis Using Neural Networks

This project applies **deep learning techniques** to perform sentiment analysis on a dataset of IMDB movie reviews, using a **Recurrent Neural Network (RNN)**—specifically a **Gated Recurrent Unit (GRU)**—to classify reviews as positive or negative. The primary research question is: *Can a neural network model effectively predict the sentiment of reviews in the IMDB subset from the UCI dataset?*

## 🧪 Data Exploration and Preparation

The dataset consists of 1,000 labeled reviews, which are cleaned, de-duplicated, and processed. Data preparation includes **tokenization**, **stopword removal**, and **lemmatization**, resulting in a vocabulary size of 2,862. The reviews are then split into training (70%), validation (15%), and test (15%) sets. The sequences are padded to a maximum length of 41 tokens based on statistical analysis of review lengths. Reviews are encoded using TensorFlow's `Tokenizer`, and padding ensures all inputs to the network are of uniform size.

## 🧠 Neural Network Architecture

The model is constructed using Keras and includes:
- An **embedding layer** that converts tokens into dense vectors.
- A **GRU layer** with 81 units and L2 regularization.
- **Batch normalization** and **dropout** layers to reduce overfitting.
- A **dense sigmoid output layer** for binary classification.

The model’s architecture was fine-tuned using **Keras Tuner's Hyperband**, with Nadam as the optimizer and binary cross-entropy as the loss function. Key hyperparameters such as neuron count, dropout rate, and learning rate were optimized through automated search.

## 📊 Evaluation and Results

The model achieved:
- **Training accuracy**: 97.13%
- **Validation accuracy**: 81.33%
- **Test accuracy**: 65.33%

While the model performs well on training and validation sets, the gap in test accuracy indicates some **overfitting**. Early stopping and dropout were implemented to mitigate this, but further adjustments—such as adding more diverse training data or regularization—are recommended.

## 🚀 Course of Action and Applications

To improve generalization, the model could be enhanced with more data, additional regularization techniques, or alternative architectures like LSTM or bidirectional RNNs. Once refined, this sentiment model could be deployed to analyze customer reviews in healthcare settings, such as evaluating patient feedback at WGU Hospital to improve service delivery.

This project demonstrates a full deep learning pipeline—from preprocessing text data to deploying a neural network for real-world sentiment classification.
