from utils import *
from algorithm import Algorithm


class SpectralResidual(Algorithm):
    def __init__(self,
                 fourier_window_size : int = 3,
                 series_window_size : int = 5,
                 score_window_size : int = 21) -> None:
        """
        Initialize the SR class.

        Parameters:
        fourier_window_size (int): The size of the window for the Fourier average. (w in the paper)
        series_window_size (int): The size of the series window. (m in the paper)
        score_window_size (int): The size of the score window. (z in the paper)
        """
        
        self.fourier_window_size = fourier_window_size
        self.series_window_size = series_window_size
        self.score_window_size = score_window_size

    def saliency_map(self, series):
        """
        Calculates the saliency map of a given time series.

        Parameters:
        - series: numpy array
            The input time series.

        Returns:
        - saliency_map: numpy array
            The saliency map of the input time series.
        """

        # Perform Fourier transform on the input series
        fourier = np.fft.fft(series)
        amplitude = np.abs(fourier)

        # Calculate the logarithm of the amplitude
        log_amplitude = np.log(amplitude)

        # Calculate the moving average of the logarithm of the amplitude
        average_log_amplitude = moving_mean(log_amplitude, self.fourier_window_size)

        # Calculate the spectral residual
        spectral_residual = log_amplitude - average_log_amplitude

        # Calculate the exponential of the spectral residual
        exp_sr = np.exp(spectral_residual)

        # Keep the phase but change the amplitude
        fourier.real = fourier.real * exp_sr / amplitude
        fourier.imag = fourier.imag * exp_sr / amplitude

        # Perform inverse Fourier transform 
        saliency_map = np.fft.ifft(fourier)

        # Return the absolute value to obtain the saliency map
        return np.abs(saliency_map)


    def compute_scores(self, series):
        """
        Compute the anomaly scores for a given time series.

        Parameters:
        series (array-like): The input time series.

        Returns:
        array-like: The anomaly scores for the time series.
        """

        extended_series = extend(series, self.series_window_size, self.series_window_size)
        spectral_residual = self.saliency_map(extended_series)[: len(series)]

        mean = moving_mean(spectral_residual, self.score_window_size)
        score = np.abs(spectral_residual - mean) / mean

        return score

    def evaluate(self, data, fourier_window_size: int = None, series_window_size: int = None, score_window_size: int = None, *args, **kwargs):
        """
        Evaluate the given data using the specified window sizes for Fourier transformation, series window, and score window.

        Args:
            data (list): The input data to be evaluated.
            fourier_window_size (int, optional): The window size for Fourier transformation. Defaults to None.
            series_window_size (int, optional): The window size for series. Defaults to None.
            score_window_size (int, optional): The window size for scoring. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The computed scores for the input data.
        """
        if fourier_window_size is not None:
            self.fourier_window_size = fourier_window_size
        if series_window_size is not None:
            self.series_window_size = series_window_size
        if score_window_size is not None:
            self.score_window_size = score_window_size

        scores = self.compute_scores(data)
        return scores