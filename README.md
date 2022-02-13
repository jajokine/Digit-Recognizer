# Digit-Recognizer
MITx - MicroMasters Program on Statistics and Data Science - Machine Learning with Python 

Second Project - Non-Linear Classifier for Digit Analysis

The second project for the MIT MicroMasters Program course on Machine Learning with Python was aimed in discovering
non-linear ways to classify digits using the famous MNIST (Mixed National Institute of Standards and Technology) database.

The MNIST database contains binary images of handwritten digits collected from among Census Bureau employees and high school
students, and it is commonly used to train image processing systems. The database contains 60,000 training images and
10,000 testing images; All of which have been size-normalized and centered in a fixed size of 28 x 28 pixels.

The goal was to first implement and understand why linear methods such as the linear regression aren't adequate for a classification task, but rather
non-linear ways, such as the multinomial logistic regression (Softmax regression) are more appropriate. Furthermore,  by adjusting hyperparameters,
and feature engineering the data into low-dimensional features with Principal Components Analysis (PCA) or high-dimensional by implementing kernel methods such
as Polynomial and Gaussian RBF kernels, it is possible to achieve even better results.

Additional helper functions were given to complete the project in two weeks of time.

DATASET

The function call get_MNIST_data() returns the following Numpy arrays:

    - train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are
    the raw pixel values flattened out into a vector of length 28^2 = 784. The pixel values are float values
    between 0 and 1 (0 for black, 1 for white, and various shades of gray in-between.
    
    - train_y : The labels for each training datapoint that are the digit numbers for each image (i.e. a number between 0-9).
    
    - test_x : A matrix of the test data, formatted the same way as the training data.
    
    - test_y : The labels for the test data, formatted the same wat as the training data.
