import numpy as np
import yfinance as yf


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
    close: np.array = df["Close"].to_numpy()
    volume: np.array = df["Volume"].to_numpy()
    
    return dates, close, volume


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
    """_summary_

    Args:
        time_series (np.array): _description_

    Returns:
        np.array: _description_
    """
    return low_pass(time_series, period)


def EMA(time_series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (EMA).
    
    EMA_t = α * X_t + (1 - α) * EMA_{t-1}
        
    where:
        α = 2 / (period + 1)
        X_t = value at time t
    
    The first EMA is initialized as the Simple Moving Average (SMA)
    of the first `period` values:
        EMA_{period-1} = mean(X_0, ..., X_{period-1})
    
    Args:
        time_series (np.ndarray): Time series of prices/values.
        period (int): Smoothing period.
    
    Returns:
        np.ndarray: Series with EMA values (NaN before the first valid EMA).
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
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    Formulas:
        MACD line      = EMA_fast - EMA_slow
        Signal line    = EMA(MACD line, period=signal)
        Histogram      = MACD line - Signal line

    Args:
        close (np.ndarray): Array of closing prices.
        fast (int, optional): Period for the fast EMA. Default is 12.
        slow (int, optional): Period for the slow EMA. Default is 26.
        signal (int, optional): Period for the signal line EMA. Default is 9.

    Returns:
        tuple of np.ndarray: 
            - macd_line: The MACD line (fast EMA - slow EMA).
            - signal_line: The signal line (EMA of the MACD line).
            - histogram: Difference between the MACD line and the signal line.
    
    Notes:
        - The function pads the shorter arrays to align lengths.
        - The histogram is useful to visualize the strength and direction of momentum.
        - Standard MACD parameters are fast=12, slow=26, signal=9.
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
    Calculate the Relative Strength Index (RSI) of a time series.

    Formulas:
        RS  = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    Args:
        close (array-like): Array of closing prices.
        period (int, optional): Look-back period for RSI calculation. Default is 14.

    Returns:
        np.ndarray: Array of RSI values corresponding to each closing price.
                    The first 'period' values are set to 50 (neutral).

    Interpretation:
        - RSI > 70 → Overbought (Sell signal)
        - RSI < 30 → Oversold (Buy signal)
        - RSI between 30 and 70 → Neutral / Hold
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
