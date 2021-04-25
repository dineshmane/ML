# ML
This repository contains the 

### 01 Machine Learning in Python - Predictive Modelling
- The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 
- The classification goal is to predict if the client will subscribe a term deposit (variable y).
- Implemented classification models like Decision tree, KNN, Random forest, Ada-boost, SVM, ANN using Scikit-learn package and evaluated their performance

### 02 Ensemble Super Learner
- Implemented and evaluated the Super Learner Classifier in python using Sklearn models 
- Evaluated the impact of adding original descriptive features at the stack layer 
- Added functionality for interrogating performance and correlation of base models 
- Compared alternative stack layer approaches
- Performed and interpreted grid search

### 03 Deep Learning With Keras
- A popular demonstration of the capability of deep learning techniques is object recognition in image data.
- Implemented neural network with multiple hidden layers with ReLU and softmax activation functions. A softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.
- Implemented simple CNN for MNIST that demonstrates how to use all of the aspects of a modern CNN implementation, including Convolutional layers, Pooling layers and Dropout layers.

### 04 Cifar
- The CIFAR-10 data consists of 60,000 (32×32) colour images in 10 classes, with 6000 images per class. There are 50,000 training images and 10,000 test images in the official data. 
- Implemented Artificial Neural network and Convoluted Neural network to learn features at various levels of abstraction.

### 05 k Armed Bandit
- This is a classic reinforcement learning problem that exemplifies the exploration-exploitation trade-off dilemma
- In the problem, each machine provides a random reward from a probability distribution specific to that machine. 
- The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls. The crucial trade-off the gambler faces at each trial is between "exploitation" of the machine that has the highest expected payoff and "exploration" to get more information about the expected payoffs of the other machines.

### 06 Sequential Models
- This notebook talks about the simple code snippets for Simple RNN, GRU and LSTM implementation in Keras. 
- These three models’ performance is compared on IMDB movie dataset

### 07 Market Segmentation
- Customer segmentation is a method of dividing customers into groups or clusters on the basis of common characteristics. 
- The market researcher can segment customers into the B2C model using various customer's demographic characteristics such as occupation, gender, age, location, and marital status. 
- RFM (Recency, Frequency, Monetary) analysis is a behaviour-based approach grouping customers into segments. 
- It groups the customers on the basis of their previous purchase transactions. How recently, how often, and how much did a customer buy. 
- RFM filters customers into various groups for the purpose of better service. 
- It helps managers to identify potential customers to do more profitable business.

### 08 Generative Adversarial Networks (GANs)
- Implemented GANs on CIFAR10 dataset in python using PyTorch 
- Defined Generator and Discriminator module of a neural network containing a sequence of modules (convolutions, full connections, etc.) 
- Used Adam optimizer for Generator as well as Discriminator

### 09 TensorFlow Getting Started Pluralsight
- Built simple regression model and advanced deep neural network models in TensorFlow, and utilized additional libraries like Keras, TFLearn that make development even easier. 
- Explored TensorFlow with neural networks in general, and specifically with powerful deep neural networks. 

### 10 UFO Sightings ALgorithm Sagemaker:AcloudGuru
- The goal of this notebook is to build out models to use for predicting the legitimacy of a UFO sighting using the Sagemaker's built-in Linear Learner algorithm.
- We loaded dataset onto Notebook instance memory from S3, pre-processed the dataset, create-train-deployed the model in Sagemaker
- Created Lambda function and called invokeEndpoint() method for real time predications/inferences.

### 11 Lego Classifier Neural Net:AcloudGuru
- We also have a collection of 600 photos of various LEGO bricks from different angles. And all the images are stored into data arrays for easier loading into the notebook. These are stored in the lego-simple-train-images.npy and lego-simple-test-images.npy files
- Created a simple, deep learning, neural network classifier model. Trained the model using the photo data and cross-checked if it correctly predicts the type of a brick from a supplied test image.
