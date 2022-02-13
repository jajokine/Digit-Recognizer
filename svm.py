# Functions for Support Vector Machine (SVM) linear classifiers

import numpy as np
from sklearn.svm import LinearSVC

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classification

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    
    clf = LinearSVC(C = 0.1, random_state = 0)
    clf.fit(train_x, train_y)
    pred_test_y = clf.predict(test_x)
    
    return pred_test_y

  
def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classification using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    
    clf = LinearSVC(C = 0.1, random_state=0)
    clf.fit(train_x, train_y)
    pred_test_y = clf.predict(test_x)
    
    return pred_test_y

def compute_test_error_svm(test_y, pred_test_y):
  """
  Computes the test error from the SVM functions
  
  Args:
      test_y - (n, ) NumPy array containing the labels (int) for each test data point
      pred_test_y - (m, ) NumPy array containing the predicted labels (int) for each test data point
  
  Returns:
      The fraction of labels that don't match the target labels
    """
    
    return 1 - np.mean(pred_test_y == test_y)