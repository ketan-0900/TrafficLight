```
Project  Report on Traffic Light Dectector
Index:
1.	Introduction
2.	Background
3.	Underlying Concepts 
4.	Specifications
Introduction:
This project implements an AI to identify which traffic sign appears in a photograph. It is designed with Python. It has a comprehensive and large standard library that has automatic memory management and dynamic features. The ones used in this project are OpenCV, TensorFlow and Scikit-learn. 
The project was tested and it turned out to be 96.06% accurate.

```
```
$ python traffic.py gtsrb
Train on 15984 samples
Epoch 1/10
15984/15984 [==============================] - 10s 623us/sample - loss: 2.8565 - accuracy: 0.3022
Epoch 2/10
15984/15984 [==============================] - 8s 510us/sample - loss: 1.3484 - accuracy: 0.5951
Epoch 3/10
15984/15984 [==============================] - 8s 531us/sample - loss: 0.8283 - accuracy: 0.7494
Epoch 4/10
15984/15984 [==============================] - 12s 736us/sample - loss: 0.5758 - accuracy: 0.8270
Epoch 5/10
15984/15984 [==============================] - 12s 744us/sample - loss: 0.4241 - accuracy: 0.8725
Epoch 6/10
15984/15984 [==============================] - 10s 602us/sample - loss: 0.3391 - accuracy: 0.8956
Epoch 7/10
15984/15984 [==============================] - 10s 620us/sample - loss: 0.3102 - accuracy: 0.9103
Epoch 8/10
15984/15984 [==============================] - 11s 668us/sample - loss: 0.2747 - accuracy: 0.9207
Epoch 9/10
15984/15984 [==============================] - 10s 614us/sample - loss: 0.2208 - accuracy: 0.9362
Epoch 10/10
15984/15984 [==============================] - 8s 528us/sample - loss: 0.1961 - accuracy: 0.9418
10656/10656 - 2s - loss: 0.1392 - accuracy: 0.9606
```
```
Underlying Concepts:
Gradient Descent
Algorithm for minimizing loss when training a neural network.
•	Calculate the gradient based on all data points: direction that will lead to decreasing loss.
But it would be very complex and time taking.
Alternatives:-
Stochastic Gradient Descent
Start with a random choice of weights. 
• Repeat:
 • Calculate the gradient based on one data point: direction that will lead to decreasing loss.
 • Update weights according to the gradient.
Mini-Batch Gradient Descent
.  • Calculate the gradient based on one small batch: direction that will lead to decreasing loss. 
Multilayer neural network 
Artificial neural network with an input layer, an output layer, and at least one hidden layer. (Instead of using perceptron, we use multilevel neural network because perceptron is only capable of learning linearly separable decision boundary.)
Backpropagation 
Algorithm for training neural networks with hidden layers.
• Start with a random choice of weights. 
• Repeat: 
• Calculate error for output layer. 
• For each layer, starting with output layer, and moving inwards towards earliest hidden layer: 
• Propagate error back one layer. 
• Update weights
Deep neural networks 
Neural network with multiple hidden layers.

Dropout 
Temporarily removing units — selected at random — from a neural network to prevent over-reliance on certain units. (to avoid overfitting)
Computer vision 
Computational methods for analyzing and understanding digital images
Image convolution 
Applying a filter that adds each pixel value of an image to its neighbors, weighted according to a kernel matrix.
Pooling 
Reducing the size of an input by sampling from regions in the input. This is done to avoid overfitting.
Max-pooling 
Pooling by choosing the maximum value in each region. 
Flattening
Flatten our pooled feature map into a column like in the image below. The reason we do this is that we're going to need to insert this data into an artificial neural network later on.
Background:
As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.
In this project, TensorFlow is used to build a neural network to classify road signs based on an image of those signs. To do so, a  labeled dataset: a collection of images that have already been categorized by the road sign represented in them, is required.
Several such data sets exist, but for this project, German Traffic Sign Recognition Benchmark (GTSRB) dataset is used, which contains thousands of images of 43 different kinds of road signs.
Specifications:
•	The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
o	 data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
o	 OpenCV-Python module (cv2) is used to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
o	The function returns a tuple (images, labels). Images are be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size.  labels is a list of integers, representing the category number for each of the corresponding images in the images list.
o	The function is platform-independent: that is to say, it works regardless of operating system. 
•	The get_model function returns a compiled neural network model.
o	 The input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
o	The output layer of the neural network has NUM_CATEGORIES units, one for each of the traffic sign categories.
```


