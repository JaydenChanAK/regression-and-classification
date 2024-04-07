'''
This file contains useful helper functions for regression and classification models.
'''

# ===== Packages =====
import numpy as np
import matplotlib.pyplot as plt

# ===== Helper Functions ======
def load_data(filename):
    '''
    Processes the dataset into arrays. Assumes all columns are used and targets/labels are the last column.
    
    Args:
        filename (str) : The dataset (.csv).
    Returns:
        categories (ndarray)    : An array containing the categories of the dataset.
        x (ndarray Shape (m,n)) : An array containing the featuries of the dataset, excluding the last column.
        y (ndarray Shape (m,1)) : An array containing the labels of the dataset. 
    '''
    # Reads the first line to determine the number of columns
    with open(filename, 'r') as file:
        num_columns = int(file.readline().strip())
    
    dataset = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=str)
    
    categories = dataset[0, :num_columns-1]
    x = dataset[1:, :num_columns-1].astype(float)
    y = dataset[1:, num_columns-1].astype(float)
    
    return categories, x, y

def load_data_housingPrices(filename, start_col):
    '''
    Processes the dataset into arrays. Works only for "Housing.csv".
    
    Args:
        filename (str)  : The dataset (.csv).
        start_col (int) : Starting column for loading data.
    Returns:
        categories (ndarray)    : An array containing the categories of the dataset.
        x (ndarray Shape (m,n)) : An array containing the featuries of the dataset, excluding the last column.
        y (ndarray Shape (m,1)) : An array containing the labels of the dataset. 
    '''
    
    # Reads the first line to determine the number of columns
    with open(filename, 'r') as file:
        num_columns = int(file.readline().strip())
    
    dataset = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=str)
    
    categories = dataset[0, start_col+1:num_columns-1]
    x = dataset[1:, start_col+1:num_columns-1].astype(float)
    y = dataset[1:, start_col].astype(float)
    
    return categories, x, y

def sigmoid(z):
    '''
    Computes the sigmoid function.
    
    Args:
        z (ndarray) : The input array.
    Returns:
        g (ndarray) : An array containing the sigmoid(z) with the same size as z.
    '''
    
    # Sigmoid formula
    return 1/(1+np.exp(-z))

def feature_scaling(x):
    """
    Scale features by dividing each feature by its range (max - min).
    
    Args:
        x (ndarray) : An array containing the features of the dataset.
    
    Returns:
        x_scaled (ndarray) : An array containing the scaled features of the dataset.

    """
    min_val = np.min(x, axis=0)  # Compute the minimum value of each feature
    max_val = np.max(x, axis=0)  # Compute the maximum value of each feature
    x_scaled = (x - min_val) / (max_val - min_val)  # Scale the features
    return x_scaled

def logistic_graph(x, y, category1, category2, marker_size, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    plt.plot(x[positive, category1], x[positive, category2], 'r+', label=pos_label, markersize = marker_size)
    plt.plot(x[negative, category1], x[negative, category2], 'bo', label=neg_label, markersize = marker_size)

def map_features(x):
    """
    Feature mapping function to polynomial features

    Args:
        x (ndarray) : An array containing the features of the dataset.

    Returns:
        mapped_x (ndarray) : An array containing the mapped polynomial features.
    """

    degree = 6
    num_features = x.shape[1]
    out = []
    
    for i in range(num_features):
        for j in range(i + 1):
           out.append((x[:, i]**(degree - j)) * (x[:, j]**j))
    
    return np.stack(out, axis=1)