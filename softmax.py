# Functions for Multinomial (Softmax) Regression and Gradient Descent

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i]

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: 
        X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    
    column_of_ones = np.zeros([len(X), 1]) + 1 # Creates a column vector of ones of the size of the input
    
    return np.hstack((column_of_ones, X))

  
def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    
    r = (theta.dot(X.T)) / temp_parameter 
    c = np.max(r, axis = 0) # Fixed deduction factor for numerical stability 
    H = np.exp(r - c) # Subtracting the fixed amount c from each exponent to keep the resulting number from getting too large (matrix)
    H = H / np.sum(H, axis = 0) # Dividing with the normalizing term
    
    return H

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each data point
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    
    k = theta.shape[0]  # Number of labels
    n = X.shape[0]  # Number of samples
    
    probabilities = np.clip(compute_probabilities(X, theta, temp_parameter), 1e-15, 1-1e-15)  # Clip to making sure the probabilities stay inside the boundaries
    log_probabilities = np.log(probabilities) # Log of probabilities
    
    # Using sparse matrices to handle large matrices efficiently (NumPy array of 1s and 0s)
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray() # Sparse matrix of [[y^(i)==j]] ; M[i][j] = 1 if y^(j) = 1 and 0 otherwise
    error_term = (-1/n) * np.sum(log_probabilities[M==1]) # Calculate error where M equals 1
    
    regularization_term = (lambda_factor/2) * np.linalg.norm(theta)**2  # Regularization error
    
    return error_term + regularization_term

  
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each data point
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
        A smaller temperature parameter means that there is less variance in the distribution,
        larger temperature, more variance. A larger temperature parameter makes the distribution more uniform.

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    
    k = theta.shape[0]  # Number of labels
    n = X.shape[0]  # Number of samples
    
    # Using sparse matrices to handle large matrices efficiently (NumPy array of 1s and 0s)
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n))  # Sparse matrix of [[y^(i)==j]] ; M[i][j] = 1 if y^(j) = 1 and 0 otherwise
    probabilities = compute_probabilities(X, theta, temp_parameter) # Probabilities
    gradients = (-1 / (temp_parameter * n)) * ((M - probabilities) @ X) + lambda_factor * theta  # Calculating gradients
    
    theta = theta - alpha * gradients  # Gradient descent update
    
    return theta

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    
    cost_function_progression = []
    
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for each data point
    """
    
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    
    return np.argmax(probabilities, axis=0)

  
def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3) labels (i.e. separating data set into 3 groups).

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9) for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9) for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2) for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2) for each datapoint in the test set
    """
    
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)
    
    return (train_y_mod3, test_y_mod3)

  
def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of the mod 3 labels when the classifier predicts the digit

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each data point
        theta - (k, d) NumPy array, where row j represents the parameters of the model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    
    y_pred = get_classification(X, theta, temp_parameter) # Using get_classification() for label predictions
    
    return 1 - (np.mod(y_pred, 3) == Y).mean()


def compute_test_error(X, Y, theta, temp_parameter):
  """
  Returns the error of the labels when the classifier the classifier predicts the digit
  
  Args:
      X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
      Y - (n, ) NumPy array containing the labels (a number from 0-9) for each data point
      theta - (k, d) NumPy array, where row j represents the parameters of the model for label j
      temp_parameter * the temperature parameter of softman function (scalar)
  
  Returns:
      test_error - the error rate of the classifier (scalar)
  """
    
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)


def plot_cost_function_over_time(cost_function_history):
    """
  Plots all the iterations of the cost function

  Args:
      cost_function_history - A Python list containing the cost calculated at each step of gradient descent

  Returns:
     A graph of the cost function iteration
    """

    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()
