import warnings
from typing import Callable, List, NamedTuple, Optional, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import DBSCAN
from sklearn.covariance import EmpiricalCovariance
from algorithm import Algorithm
from utils import dwt, MLE, logprobdensity, predict


class DWTMLEAD(Algorithm):
    
    def __init__(self) -> None:
        super().__init__()
    
    
    def detect_anomalies(self,
                        timeseries: np.ndarray,
                        d_max : float, 
                        level_start : int=0, 
                        level_end : int=None,
                        wavelet : str='haar',
                        epsilon : float=0.01,
                        B : int=10) -> np.ndarray:
        
        # Compute DWT
        if level_end is None:
            level_end = np.log2(len(timeseries))
        cA, cD, level = dwt(timeseries, level_start, level_end, wavelet) # Coefficients d'approximations, coefficient de détails
        
        # Setting a leaf counter 
        h = np.zeros(len(timeseries))
        
        # Set window sizes
        w = np.zeros(level_end - level_start +1)
        for i in range(level_start, level_end+1):
            w[i-level_start] = max(2, i - level_start + 1)
            
        # Sliding the windows
        D = []
        C = []
        for i in range(len(w)):
            C.append(sliding_window_view(cA[i], window_shape=w[i]))
            D.append(sliding_window_view(cD[i], window_shape=w[i]))
        
        
        x_prop = C[:len(C)-1] + D
        for x in x_prop:
            mu, sigma = MLE(x)
            p = logprobdensity(x, mu, sigma)
            z_eps = np.percentile(p, 100*(1-epsilon))
            a = predict(p, z_eps)
            for j in range(len(a)):
                if a[j] == 1:
                    h[j] += 1
        for i in range(len(h)):
            if h[i] < 2:
                h[i] = 0
        Clusters = []
        C
            
            
            
        
        
                        