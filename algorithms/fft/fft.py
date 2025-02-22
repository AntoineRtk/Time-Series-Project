from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import numpy as np

from scipy.signal import savgol_filter
from sporco.admm import bpdn

from algorithm import Algorithm

@dataclass
class LocalOutlier:
    index: int
    z_score: float

    @property
    def sign(self) -> int:
        return np.sign(self.z_score)


@dataclass
class RegionOutlier:
    start_idx: int
    end_idx: int
    score: float


@contextmanager
def nested_break():
    class NestedBreakException(Exception):
        pass

    try:
        yield NestedBreakException
    except NestedBreakException:
        pass

class FFT(Algorithm):
    
    def moving_average(self, x, window_size = 20):
        # the moving average is a convolution of the signal with a vector of 1 divided by the window size
        return np.convolve(x, np.ones(window_size) / window_size, mode = 'same')
    
    def dictionary_learning(self, data, fs):
        # code inspired from https://github.com/oudre/parcoursIA/blob/main/Session5.ipynb
        # We use a Fourier features dictionary
        D = np.zeros((len(data), 2 * int(np.floor(len(data) / 2)) + 1))
        D [:, 0] = np.ones(len(data))
        for k in range(int(np.floor(len(data) / 2))):
            f0 = fs / len(data)
            D[:, 2 * k + 1] = np.cos(2 * np.pi * f0 * (k+1) * np.arange(len(data)) / fs)
            D[:, 2 * k + 2] = np.sin(2 * np.pi * f0 * (k+1) * np.arange(len(data)) / fs)
        lmbda = 2
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                            'RelStopTol': 1e-8, 'AutoRho': {'RsdlTarget': 1.0}})
        b = bpdn.BPDN(D, data.reshape(-1, 1), lmbda, opt)
        z = b.solve()
        return (D @ z).reshape(-1,)
    
    def reduce_parameters(self, f: np.ndarray, k: int) -> np.ndarray:
        """
        :param f: fourier transform
        :param k: number of parameters to use
        :return: fourier transform value reduced to k parameters (including the zero frequency term)
        """
        transformed = f.copy()
        if k == 1:
            transformed[1:] = .0
        else:
            transformed[k:-(k - 1)] = 0
        return transformed


    def calculate_local_outlier(self, data: np.ndarray, k: int, c: int, threshold: float) -> List[LocalOutlier]:
        """
        :param data: input data (1-dimensional)
        :param k: number of parameters to be used in IFFT
        :param c: lookbehind and lookahead size for neighbors
        :param threshold: outlier threshold
        :return: list of local outliers
        """
        n = len(data)
        k = max(min(k, n), 1)
        # Fourier transform of data
        y = self.reduce_parameters(np.fft.fft(data), k)
        f2 = np.real(np.fft.ifft(y))
        # difference of actual data value and the fft fitted curve
        so = np.abs(f2 - data)
        # average difference
        mso = np.mean(so)

        scores = []
        score_idxs = []
        for i in range(n):
            # if the difference at particular point > the average difference
            if so[i] > mso:
                # average value of 'c' neighbors on both sides
                nav = np.average(data[max(i - c, 0):min(i + c, n - 1)])
                # add the local difference (difference of the point and its neighbors) to the collection
                scores.append(data[i] - nav)
                # add the index of suspected outlier to the collection
                score_idxs.append(i)
        scores = np.array(scores)

        #  find average and standard deviation of local difference
        ms = np.mean(scores)
        sds = np.std(scores)

        results = []
        for i in range(len(scores)):
            # calculate the difference between local difference and mean of local difference and divide this by standard
            # deviation of local difference
            z_score = (scores[i] - ms) / sds
            # declare this as an outlier if greater than threshold
            if abs(z_score) > threshold * sds:
                index = score_idxs[i]
                results.append(LocalOutlier(index, z_score))
        return results


    def calculate_region_outlier(self, l_outliers: List[LocalOutlier], max_region_length: int, max_local_diff: int) -> List[
        RegionOutlier]:
        """
        :param l_outliers: list of local outliers with their z_score
        :param max_region_length: maximum outlier region length
        :param max_local_diff: maximum difference between two closed oppositely signed outliers
        :return: list of region outliers
        """

        def distance(a: int, b: int) -> int:
            if a > b:
                h = a
                a = b
                b = h
            return l_outliers[b].index - l_outliers[a].index

        regions = []
        i = 0
        n_l = len(l_outliers) - 1
        while i < n_l:
            s_sign = l_outliers[i].sign
            s_sign2 = l_outliers[i + 1].sign
            if s_sign != s_sign2 and distance(i, i + 1) <= max_local_diff:
                i += 1
                start_idx = i
                for i in range(i + 1, n_l):
                    e_sign = l_outliers[i].sign
                    e_sign2 = l_outliers[i + 1].sign
                    if s_sign2 == e_sign and distance(start_idx, i) <= max_region_length \
                            and e_sign != e_sign2 and distance(i, i + 1) <= max_local_diff:
                        end_idx = i
                        regions.append(RegionOutlier(
                            start_idx=start_idx,
                            end_idx=end_idx,
                            score=np.mean([abs(l.z_score) for l in l_outliers[start_idx: end_idx + 1]])
                        ))
                        i += 1
                        break
                i = start_idx
            else:
                i += 1

        return regions


    def evaluate(self, data: np.ndarray,
                         ifft_parameters: int = 5,
                         local_neighbor_window: int = 21,
                         local_outlier_threshold: float = .6,
                         max_region_size: int = 50,
                         max_sign_change_distance: int = 10,
                         preprocessing = 0,
                         fs = 0,
                         **args) -> np.ndarray:
        """
        :param data: input time series
        :param ifft_parameters: number of parameters to be used in IFFT
        :param local_neighbor_window: centered window of neighbors to consider for z_score calculation
        :param local_outlier_threshold: outlier threshold in multiples of sigma
        :param max_region_size: maximum outlier region length
        :param max_sign_change_distance: maximum difference between two closed oppositely signed outliers
        :param preprocessing: preprocess the data with 1. moving average, 2. dictionary learning, 3. Savitzky-Golay filter
        :return: anomaly scores (same shape as input)
        """
        if(preprocessing):
            if(preprocessing == 1):
                data = self.moving_average(data)
            if(preprocessing == 2):
                data = self.dictionary_learning(data, fs)
            else:
                data = savgol_filter(data, len(data) // 34, 5) # dividing by 34 is arbitrary, it did show good results in our experiments
                
        neighbor_c = local_neighbor_window // 2
        #print(ifft_parameters, neighbor_c, local_outlier_threshold, max_region_size, max_sign_change_distance)
        local_outliers = self.calculate_local_outlier(data, ifft_parameters, neighbor_c, local_outlier_threshold)
        #print(f"Found {len(local_outliers)} local outliers")
        
        regions = self.calculate_region_outlier(local_outliers, max_region_size, max_sign_change_distance)
        #print("Regions: ", regions)
        
        # broadcast region scores to data points
        anomaly_scores = np.zeros_like(data)
        for reg in regions:
            start_local = local_outliers[reg.start_idx]
            end_local = local_outliers[reg.end_idx]
            anomaly_scores[start_local.index:end_local.index + 1] = [reg.score] * (end_local.index - start_local.index + 1)

        return anomaly_scores