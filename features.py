# Functions for Feature Engineering - Principal Components Analysis (PCA) and Cubic Features

import numpy as np
import matplotlib.pyplot as plt


def center_data(X):
    """
    Returns a centered version of the data, where each feature now has mean = 0

    Args:
        X - n x d NumPy array of n data points each with d features

    Returns:
        - (n, d) NumPy array X' where for each i = 1, ..., n and j = 1, ..., d:
        X'[i][j] = X[i][j] - means[j]       
        - (d, ) NumPy array with the columns means

    """
    
    feature_means = X.mean(axis=0)
    
    return (X - feature_means), feature_means


def principal_components(centered_data):
    """
    Returns the principal component vectors of the data, sorted in decreasing order of eigenvalue magnitude.
    This function first calculates the covariance matrix and then finds its eigenvectors.

    Args:
        centered_data - n x d NumPy array of n data points each with d features

    Returns:
        d x d NumPy array whose columns are the principal component directions sorted
        in descending order by the amount of variation each direction (these are
        equivalent to the d eigenvectors of the covariance matrix sorted in descending
        order of eigenvalues, so the first column corresponds to the eigenvector with
        the largest eigenvalue)
    """
    
    scatter_matrix = np.dot(centered_data.T, centered_data)
    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix) # Getting eigen values and vectors
    
    idx = eigen_values.argsort()[::-1]  # Sorting eigenvectors by magnitude with the reverse indexing
    eigen_values = eigen_values[idx]
    pcs = eigen_vectors[:, idx]
    
    return pcs

  
def project_onto_PC(X, pcs, n_components, feature_means):
    """
    Given principal component vectors pcs = principal_components(X)
    this function returns a new data array in which each sample in X
    has been projected onto the first n_components principal components
    
    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        pcs - (n, ) NumPy array of principal comonents / eigenvectors
        n_components - Number of principal components (scalar)
        feature_means - (d, ) NumPy array with the columns means
    
    Returns:
        Projected data
    """
    
    X_centered, feature_means = center_data(X) # Centering data
    
    V_n = pcs[:, range(n_components)] # First n_components columns of the eigenvectors
    
    projected_data = X_centered @ V_n # Projecting data onto the principal components
    
    return projected_data

  
def plot_PC(X, pcs, labels, feature_means):
    """
    Given the principal component vectors as the columns of matrix pcs,
    this function projects each sample in X onto the first two principal components
    
    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        pcs - (n, ) Numpy array of principal components / eigenvectors
        labels = a numpy array containing the digits corresponding to each image in X
    
    Returns:
        A scatterplot where point are marked with the digit depicted in the corresponding image
    """
    
    pc_data = project_onto_PC(X, pcs, n_components=2, feature_means=feature_means)
    
    text_labels = [str(z) for z in labels.tolist()]
    
    fig, ax = plt.subplots()
    ax.scatter(pc_data[:, 0], pc_data[:, 1], alpha=0, marker=".")
    
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i, 0], pc_data[i, 1]))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    
    plt.show()


def reconstruct_PC(x_pca, pcs, n_components, X, feature_means):
    """
    Given the principal component vectors as the columns of matrix pcs,
    this function reconstructs a single image from its principal component
    representation, x_pca
    
    Args:
        x_pca - principal component representation of the original data
        pcs - (n, ) NumPy array of eigenvectors
        n_components - Number of principal components (scalar)
        X - (n, d) NumPy array of the original data to which PCA was applied to get pcs
        feature_means - - (d, ) NumPy array with the columns means
    
    Returns:
        The reconstructed version of the digits using the pca
    """
    
    x_reconstructed = np.dot(x_pca, pcs[:, range(n_components)].T) + feature_means
    
    return x_reconstructed


def cubic_features(X):
    """
    Returns a new dataset with features given by the mapping
    which corresponds to the cubic kernel
    [1,x_1^3,x_2^3,sqrt(3)*x_1^2*x_2,sqrt(3)*x_1*x_2^2, sqrt(3)*x_1^2,sqrt(3)*x_2^2,sqrt(6)*x_1*x_2,sqrt(3)*x_1,sqrt(3)*x_2]
    """
    n, d = X.shape  # dataset size, input dimension
    X_withones = np.ones((n, d + 1))
    X_withones[:, :-1] = X
    new_d = 0  # dimension of output
    new_d = int((d + 1) * (d + 2) * (d + 3) / 6)

    new_data = np.zeros((n, new_d))
    col_index = 0
    for x_i in range(n):
        X_i = X[x_i]
        X_i = X_i.reshape(1, X_i.size)

        if d > 2:
            comb_2 = np.matmul(np.transpose(X_i), X_i)

            unique_2 = comb_2[np.triu_indices(d, 1)]
            unique_2 = unique_2.reshape(unique_2.size, 1)
            comb_3 = np.matmul(unique_2, X_i)
            keep_m = np.zeros(comb_3.shape)
            index = 0
            for i in range(d - 1):
                keep_m[index + np.arange(d - 1 - i), i] = 0

                tri_keep = np.triu_indices(d - 1 - i, 1)

                correct_0 = tri_keep[0] + index
                correct_1 = tri_keep[1] + i + 1

                keep_m[correct_0, correct_1] = 1
                index += d - 1 - i

            unique_3 = np.sqrt(6) * comb_3[np.nonzero(keep_m)]

            new_data[x_i, np.arange(unique_3.size)] = unique_3
            col_index = unique_3.size

    for i in range(n):
        newdata_colindex = col_index
        for j in range(d + 1):
            new_data[i, newdata_colindex] = X_withones[i, j]**3
            newdata_colindex += 1
            for k in range(j + 1, d + 1):
                new_data[i, newdata_colindex] = X_withones[i, j]**2 * X_withones[i, k] * (3**(0.5))
                newdata_colindex += 1

                new_data[i, newdata_colindex] = X_withones[i, j] * X_withones[i, k]**2 * (3**(0.5))
                newdata_colindex += 1

                if k < d:
                    new_data[i, newdata_colindex] = X_withones[i, j] * X_withones[i, k] * (6**(0.5))
                    newdata_colindex += 1

    return new_data
