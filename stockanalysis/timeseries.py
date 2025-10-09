import numpy as np
import yfinance as yf


class TimeSeries:
    
    def __init__(self, values: np.array, window_size=3) -> None:
        self.values = values
        self.smooth_values = low_pass(values, window_size)
        self.first_derivative = np.gradient(self.smooth_values)
        self.second_derivative = np.gradient(self.first_derivative)


def get_time_series(ticker: yf.Ticker, period: str, interval: str) -> tuple[any]:
    """
    Returns the time series data (dates, closing prices, and volumes) 
    for a given stock ticker from Yahoo Finance.

    Args:
        ticker (yf.Ticker): Ticker with the stock ticker symbol (e.g., "AAPL" for Apple).
        period (str): Data period to download (e.g., "30d", "1y", "max").
        interval (str): Data interval (e.g., "1d", "1h", "5m").

    Returns:
        tuple:
            - dates (pandas.DatetimeIndex): The dates corresponding to the time series.
            - close (numpy.ndarray): Closing prices for each date.
            - volume (numpy.ndarray): Trading volumes for each date.
    """
    df = ticker.history(period=period, interval=interval)
    
    dates = df.index
    
    open: np.array = df["Open"].to_numpy()
    high: np.array = df["High"].to_numpy()
    low: np.array = df["Low"].to_numpy()
    
    close: np.array = df["Close"].to_numpy()
    volume: np.array = df["Volume"].to_numpy()
    
    return dates, open, high, low, close, volume


def low_pass(time_series: np.array, window_size=3) -> np.array:
    """
    Applies a low-pass filter to a 1D time series using a simple moving average (SMA).
    Near the boundaries, the window is reduced to fit the available data instead of 
    returning NaN or truncating the output.

    Args:
        time_series (numpy.ndarray): Input time series data as a 1D NumPy array.
        window_size (int, optional): Maximum size of the moving average window.
            Must be a positive integer. Default is 3.

    Returns:
        numpy.ndarray: Smoothed time series of the same length as the input.
                       At the edges, the effective window is smaller.
    """
    n = len(time_series)
    result = np.empty(n, dtype=float)

    half = window_size // 2
    for i in range(n):
        # Define start and end of the window, clipped to series bounds
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.mean(time_series[start:end])
    
    return result


def linear_regression(time_series: np.ndarray) -> tuple[float, float]:
    """
    Performs simple linear regression on a time series.

    The function fits a straight line of the form y = m * x + n, 
    where `x` represents the time index and `y` the time series values.

    Args:
        time_series (numpy.ndarray): Input time series data as a 1D NumPy array.

    Returns:
        tuple:
            - m (float): The slope of the fitted line.
            - n (float): The intercept of the fitted line.
    """
    k = len(time_series)
    x = np.arange(k)
    y = time_series

    sumX = np.sum(x)
    sumY = np.sum(y)
    sumXY = np.sum(x * y)
    sumX2 = np.sum(x * x)

    m = (k * sumXY - sumX * sumY) / (k * sumX2 - sumX**2)
    n = (sumY - m * sumX) / k

    return m, n


def SMA(time_series: np.array, period: int) -> np.array:
    """
    Computes the Simple Moving Average (SMA) of a 1D time series.

    This function is a wrapper around `low_pass`, applying a moving 
    average filter with a fixed window size (`period`). The SMA is a 
    common technical analysis tool that smooths short-term fluctuations 
    to highlight longer-term trends.

    At the edges of the series, the window is automatically reduced to 
    fit the available data (instead of padding or discarding values).

    Args:
        time_series (numpy.ndarray): Input time series data as a 1D NumPy array.
        period (int): Size of the moving average window. Must be a positive integer.

    Returns:
        numpy.ndarray: Array of the same length as `time_series`, containing 
        the smoothed values after applying the SMA.
    
    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> SMA(data, 3)
        array([1.5, 2.0, 3.0, 4.0, 4.5])
    """
    return low_pass(time_series, period)


def EMA(time_series: np.ndarray, period: int) -> np.ndarray:
    """
    Computes the Exponential Moving Average (EMA) of a 1D time series.

    The EMA applies exponentially decreasing weights to past values, giving
    more importance to recent data compared to the Simple Moving Average (SMA).

    Formula:
        EMA_t = α * X_t + (1 - α) * EMA_{t-1}

    where:
        α = 2 / (period + 1)
        X_t = value at time t

    Initialization:
        The first EMA value (at index period-1) is set to the SMA of the 
        first `period` values.

    Args:
        time_series (numpy.ndarray): Input time series data.
        period (int): Look-back period for smoothing. Must be a positive integer.

    Returns:
        numpy.ndarray: Array of EMA values with the same length as the input.
                       Indices before the first valid EMA are set to NaN.

    Example:
        >>> import numpy as np
        >>> data = np.array([10, 11, 12, 13, 14, 15])
        >>> EMA(data, 3)
        array([nan, nan, 11.0, 12.0, 13.0, 14.0])
    """
    alpha = 2 / (period + 1)
    
    ema = np.full_like(time_series, np.nan, dtype=np.float64)
    
    if len(time_series) < period:
        return ema
    
    ema[period-1] = np.mean(time_series[:period])
    for i in range(period, len(time_series)):
        ema[i] = time_series[i] * alpha + ema[i-1] * (1 - alpha)

    return ema

def MACD(close: np.array, fast=12, slow=26, signal=9):
    """
    Computes the Moving Average Convergence Divergence (MACD) indicator.

    The MACD is a momentum oscillator that measures the relationship between 
    two EMAs of a time series (typically closing prices).

    Formulas:
        - MACD line   = EMA_fast - EMA_slow
        - Signal line = EMA(MACD line, period = signal)
        - Histogram   = MACD line - Signal line

    Args:
        close (numpy.ndarray): Closing prices or time series values.
        fast (int, optional): Period for the fast EMA. Default is 12.
        slow (int, optional): Period for the slow EMA. Default is 26.
        signal (int, optional): Period for the signal line EMA. Default is 9.

    Returns:
        tuple of numpy.ndarray:
            - macd_line (np.ndarray): Difference between fast and slow EMAs.
            - signal_line (np.ndarray): EMA of the MACD line.
            - histogram (np.ndarray): MACD line minus signal line.

    Notes:
        - Standard parameters are (fast=12, slow=26, signal=9).
        - The histogram is commonly used to assess momentum shifts.

    Example:
        >>> prices = np.array([10, 11, 12, 13, 14, 15, 16])
        >>> macd_line, signal_line, hist = MACD(prices)
    """
    ema_fast = EMA(close, fast)
    ema_slow = EMA(close, slow)
    
    ema_slow = np.pad(ema_slow, (len(ema_fast)-len(ema_slow), 0), 'constant', constant_values=(ema_slow[0],))
    
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    signal_line = np.pad(signal_line, (len(macd_line)-len(signal_line), 0), 'constant', constant_values=(signal_line[0],))
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def RSI(close: np.array, period=14) -> np.array:
    """
    Computes the Relative Strength Index (RSI) of a 1D time series.

    The RSI is a momentum oscillator that measures the speed and magnitude
    of recent price changes, oscillating between 0 and 100.

    Formulas:
        RS  = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    Args:
        close (numpy.ndarray): Closing prices or time series values.
        period (int, optional): Look-back period for RSI calculation. Default is 14.

    Returns:
        numpy.ndarray: Array of RSI values. The first `period` values are
                       initialized to 50 (neutral).

    Interpretation:
        - RSI > 70 → Overbought (potential sell signal).
        - RSI < 30 → Oversold (potential buy signal).
        - 30 ≤ RSI ≤ 70 → Neutral / Hold.

    Example:
        >>> prices = np.array([10, 11, 12, 11, 10, 9, 8, 9, 10])
        >>> RSI(prices, period=5)
        array([50., 50., 50., 50., 50., 40., 30., 40., 50.])
    """
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(close, dtype=float)
    avg_loss = np.zeros_like(close, dtype=float)

    # Initial simple average
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()

    # Wilder's smoothing
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    # Robust RSI calculation to avoid divide by zero
    rsi = np.zeros_like(close, dtype=float)
    for i in range(len(close)):
        if avg_loss[i] == 0 and avg_gain[i] == 0:
            rsi[i] = 50  # neutral
        elif avg_loss[i] == 0:
            rsi[i] = 100  # max RSI
        elif avg_gain[i] == 0:
            rsi[i] = 0  # min RSI
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))

    rsi[:period] = 50  # fill first 'period' values as neutral
    return rsi
