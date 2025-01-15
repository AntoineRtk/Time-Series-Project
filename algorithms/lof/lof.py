import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.neighbors import LocalOutlierFactor
from tslearn.metrics import dtw, soft_dtw

from algorithm import Algorithm

class LOF(Algorithm):
    """
    :param n_neighbors: the number of data points to consider in the k-distance
    """
    
    def evaluate(self, timeseries, n_neighbors = 20, window_size = 101, metric = 0):
        
        def wf(x, y):
            x = np.abs(np.fft.fftshift(np.fft.fft(x)))[:len(x) // 2 + 1]
            x /= np.sum(x)
            x = np.cumsum(x)
            y = np.abs(np.fft.fftshift(np.fft.fft(y)))[:len(y) // 2 + 1]
            y /= np.sum(y)
            y = np.cumsum(y)
            return np.sum((x - y) ** 2)
    
        windows = sliding_window_view(timeseries, window_shape = window_size)
        
        if(metric == 0):
            metric = 'minkowski'
        elif(metric == 1):
            metric = dtw
        elif(metric == 2):
            metric = soft_dtw
        elif(metric == 3):
            metric = wf
        
        clf = LocalOutlierFactor(
            n_neighbors = n_neighbors,
            metric = metric
        )
        
        scores = clf.fit_predict(windows)
        
        scores = np.pad(scores, ((50), (50)), mode='constant', constant_values = 1)
        
        return -scores