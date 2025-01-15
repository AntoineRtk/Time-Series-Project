from utils import *
from algorithm import Algorithm
import stumpy

class STAMPi(Algorithm):
    def __init__(self, subsequence_length : int = 3, score_threshold : float = 0.1):
        self.subsequence_length = subsequence_length
        self.score_threshold = score_threshold
        pass
    def evaluate(self, data, subsequence_length : int = None, score_threshold : float = None, plot : bool = False, *args, **kwargs):
        """
        This function evaluates the matrix profile of a given time series data.

        Parameters:
        - data: The time series data to be evaluated.
        - subsequence_length: The length of the subsequences used for matrix profile calculation.
        - score_threshold: The threshold value for the matrix profile score to be considered as an anomaly.
        - *args, **kwargs: Additional arguments and keyword arguments.

        Returns:
        - res: The matrix profile of the time series data.
        """
        if subsequence_length is not None:
            self.subsequence_length = subsequence_length
        if score_threshold is not None:
            self.score_threshold = score_threshold
        matrix_profile = stumpy.stump(data, m=self.subsequence_length)
        res = np.zeros(len(data), dtype=np.float64)
        res[:len(matrix_profile[:, 0])] = matrix_profile[:, 0]
        res[len(matrix_profile[:, 0]):] = matrix_profile[-1, 0]
        return res
        