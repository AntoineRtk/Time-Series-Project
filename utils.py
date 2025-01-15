import pywt
import numpy as np
import csv


def dwt(data: np.ndarray, level_start: int, level_end: int = None, wavelet: str = 'haar') -> list:
    """
    Discrete Wavelet Transform
    :param data: input data
    :param level_start: start level
    :param level_end: end level
    :param wavelet: wavelet type
    :return: DWT coefficients
    """
    if level_end is None:
        level_end = np.log2(len(data))
        
    cA = data
    cA_ = []
    cD_ = []
    level_ = []
    
    for level in range(level_end): # To compute 2^n levels, we need to iterate 2^n times
        cA, cD = pywt.dwt(cA, wavelet, mode='periodic')
        
        if (level+1) >= level_start: #starts at 1 in the paper
            cA_.append(cA)
            cD_.append(cD)
            level_.append(level+1)
    return cA_, cD_, np.array(level_)

def MLE(X : np.ndarray) -> list:
    """
    Maximum Likelihood Estimation
    :param X: input data
    :return: MLE
    """
    mu = np.mean(X, axis=0)
    
    sigma = 1/(len(X)-1) * np.sum((X - mu)**2, axis=0)
    
    return mu, sigma

def logprobdensity(X : np.ndarray, mu : np.ndarray, sigma : np.ndarray) -> np.ndarray:
    p = np.zeros(len(X))
    for i in range(len(X)):
        p[i] = -0.5 * np.slogdet(2*np.pi*sigma)[1] - 0.5 * ((X[i] - mu).T).dot(np.linalg.pinv(sigma)).dot((X[i] - mu))
    return p

def predict(p, z):
    a = np.zeros(len(p))
    for i in range(len(p)):
        if p[i] < z[i]:
            a[i] = 1
        else:
            a[i] = 0
    return a


def moving_mean(tab,window_size):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    mean = np.cumsum(tab, dtype=float) # Compute mean quickly
    for i in range(window_size):
        mean[i] /= (i + 1)
    for i in range(window_size, len(tab)):
        mean[i] = (mean[i - 1] * window_size + tab[i] - tab[i - window_size]) / window_size
    return mean


def estimate_next(tab):
    """
    Extrapolates the next value by sum up the slope of the last value with previous values.
    :param values: a list or numpy array of time-series
    :return: the next value of time-series
    """

    x_n = tab[-1]
    
    # Compute the gradient of the straight lines formula (8)
    grad = [(x_n - x) / max(i,1) for (i, x) in enumerate(tab[::-1])]
    grad[0] = 0 # sum starts at 1, so the first value is 0
    
    # (9)
    next = x_n + np.cumsum(grad)

    return next


def extend(tab, k=5, m=5):
    """
    Extends a given time series by estimating the next value and repeating it for a specified number of times.

    Parameters:
    tab (list or numpy.ndarray): The input time series.
    extend_num (int): The number of times to repeat the estimated next value. Default is 5.
    forward (int): The number of steps to look ahead for estimating the next value. Default is 5.

    Returns:
    list or numpy.ndarray: The extended time series.
    """
    next = estimate_next(tab)[m]
    estimated_points = [next] * k

    if isinstance(tab, list):
        extended = next + estimated_points
    else:
        extended = np.append(tab, estimated_points)
    return extended