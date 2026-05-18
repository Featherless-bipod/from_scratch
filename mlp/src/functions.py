import numpy as np 
import pandas as pd 


def ReLu(Z):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation input matrix.

    Returns
    -------
    numpy.ndarray
        Element-wise maximum of 0 and Z.
    """
    return np.maximum(0,Z)

def softmax(Z): 
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation input matrix.

    Returns
    -------
    numpy.ndarray
        Element-wise maximum of 0 and Z.
    """
    # Subtract the max for numerical stability
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    
    # Exponentiate
    exps = np.exp(shift_Z)
    
    # Normalize
    return exps / np.sum(exps, axis=0, keepdims=True)

def one_hot(Y,num_classes=10):
    """
    Convert class labels to one-hot encoded vectors.

    Parameters
    ----------
    Y : numpy.ndarray
        Class labels (shape [batch_size,]).

    Returns
    -------
    numpy.ndarray
        One-hot encoded matrix (shape [num_classes, batch_size]).
    """
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def derv_sigmoid(Z):
    """
    Compute the derivative of the sigmoid activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values (linear combination of weights and inputs).
        Can be a scalar, vector, or matrix.

    Returns
    -------
    numpy.ndarray
        Derivative of the sigmoid function evaluated at Z.
        Shape matches the input Z.
    """
    return np.exp(-Z)/(1+np.exp(-Z))**2

def derv_ReLu(Z): 
    """
    Derivative of the ReLU function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation input matrix.

    Returns
    -------
    numpy.ndarray
        Binary mask where 1 indicates Z > 0, else 0.
    """
    return Z > 0

def get_accuracy(A,Y):
    """
    Calculate classification accuracy.

    Parameters
    ----------
    A : numpy.ndarray
        Predicted probabilities (shape [num_classes, batch_size]).
    Y : numpy.ndarray
        True labels (shape [batch_size,]).

    Returns
    -------
    float
        Accuracy between 0 and 1.
    """
    return sum(np.argmax(A,0) == Y)/Y.size #LEARN ARGMAX