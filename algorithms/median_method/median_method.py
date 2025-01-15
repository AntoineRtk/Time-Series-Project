import numpy as np
import bottleneck as bn

from algorithm import Algorithm

class MedianMethod(Algorithm):
    """
    :param np.ndarray timeseries: univariate timeseries
    :param int neighbourhood_size: number of time steps to include in the window from past and future

    example: [1, 2, 6, 4, 5] with neighbourhood_size of 1 
    move_median creates windows like this: [nan, nan, 2, 4, 4] 
    We want the indexes of the timeseries to align with the window median
    So we shift backwards like this using np.roll: [nan, 2, 4, 4, nan]
    This way we can calculate accurate differences between the timeseries data points
    and the median of their neighbourhood:
      [  1, 2, 6, 4, 5  ]
    - [nan, 2, 4, 4, nan]
    = [nan, 0, 2, 0, nan]
    """
    
    def compute_windows(self, type, timeseries, neighborhood_size):
        if type == "std":
            windows = bn.move_std(timeseries, window=neighborhood_size * 2 + 1)
        elif type == "median":
            windows = bn.move_median(timeseries, window=neighborhood_size * 2 + 1)
        return np.roll(windows, neighborhood_size)
    
    def evaluate(self, timeseries, neighborhood_size, threshold = 1, multi_scale = False):
        scores = None
        for i in range(1, 2 + 2 * multi_scale):
            median_windows = self.compute_windows("median", timeseries, neighborhood_size * i)
            std_windows = self.compute_windows("std", timeseries, neighborhood_size * i)
            dist_windows = np.absolute(timeseries - median_windows)
            if(scores is None):
                scores = dist_windows / (threshold * std_windows)
                scores = np.nan_to_num(scores)
            else:
                scores += np.nan_to_num(dist_windows / (threshold * std_windows))
        return scores / i