# Closed form solution for Linear Regression and function to compute the test error

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each datapoint
        lambda_factor - the regularization constant (scalar)
    
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    
    I = np.eye(X.shape[1])
    theta = np.linalg.inv(X.T @ X + lambda_factor * I) @ X.T @ Y

    return theta
  
def compute_test_error_linear(test_x, Y, theta):
  """
  Computes the test error from the closed form solution
  
  Args:
      test_x - (n, d) NumPy array with n datapoints each with d features
      Y - (n, ) NumPy array containing the labels (a number from 0-9) for each datapoint
      theta - (d +1, ) NumPy array containing the weights of linear regression
      
  Returns:
      The fraction of labels that don't match the target labels
  """
  
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    
    return 1 - np.mean(test_y_predict == Y)
