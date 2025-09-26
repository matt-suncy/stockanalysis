import numpy as np
import yfinance as yf
import plotext as plt


ZERO_STANDARD_DEVIATION: float = 0.01


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


def low_pass_moving_average(time_series: np.array, window_size=3) -> np.array:
    cumsum = np.cumsum(np.insert(time_series, 0, 0))  
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg


def linear_regression(time_series: np.ndarray) -> tuple[float, float]:
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


def long_term_trading(m: float, n: float):
    # Model
    action = ''
    interpretation = ''
    color_code: int = 3
    
    if m > 0:
        color_code = 2
        interpretation = 'strong trend'
        action = 'Buy'
    elif m < 0:
        color_code = 1
        interpretation = 'strong downward trend'
        action = 'Sell'
    
    if -ZERO_STANDARD_DEVIATION < m <= ZERO_STANDARD_DEVIATION:
        color_code = 3
        interpretation = 'no trend'
        action = 'Hold'

    # Print
    print(f"\033[9{color_code}m●\033[0m \033[1mLong term trading\033[0m Slope {m}\n")
    print(f"  y = mx + n")
    print(f"  m = {m}")
    print(f"  n = {n}")
    print(f"  \n\033[9{color_code}m{action}\033[0m ({interpretation})")
    print('')


def solve_decision_tree(close_derivative: float, volume_derivative: float) -> tuple[str, str, int]:
    action = ''
    interpretation = ''
    color_code: int = 3

    if close_derivative > 0 and volume_derivative > 0:
        color_code = 2
        interpretation = 'strong trend'
        action = 'Buy'
    elif close_derivative > 0 and volume_derivative < 0:
        interpretation = 'weak trend'
        action = 'Hold position'
    elif close_derivative < 0 and volume_derivative > 0:
        color_code = 1
        interpretation = 'strong downward trend'
        action = 'Sell'
    elif close_derivative < 0 and volume_derivative < 0:
        interpretation = 'weak downward trend'
        action = 'Hold position'

    return action, interpretation, color_code


def mid_term_trading(
        close_first_derivative: np.array,
        volume_first_derivative: np.array,
        close_second_derivative: np.array,
        volume_second_derivative: np.array,
        k=3
        ) -> None:
    # Model
    mean_close_first_derivative = np.sum(close_first_derivative[-k:], axis=0) / k
    mean_volume_first_derivative = np.sum(volume_first_derivative[-k:], axis=0) / k
    
    action, interpretation, color_code = solve_decision_tree(mean_close_first_derivative, mean_volume_first_derivative)

    # Print
    print(f"\033[9{color_code}m●\033[0m \033[1mMid term trading (smooth derivatives).\033[0m Considering the mean of the k={k} last derivatives\n")
    print(f"  dc/dt = {close_first_derivative[-1]}")
    print(f"  dv/dt = {volume_first_derivative[-1]}\n")
    print(f"  mean dc/dt = {mean_close_first_derivative}")
    print(f"  mean dv/dt = {mean_volume_first_derivative}\n")
    print(f"  d^2c/dt^2 = {close_second_derivative[-1]}")
    print(f"  d^2v/dt^2 = {volume_second_derivative[-1]}\n")
    print(f"  \033[9{color_code}m{action}\033[0m ({interpretation})")
    print('')


def short_term_trading(
        close_first_diff: np.array, 
        volume_first_diff: np.array, 
        close_second_diff: np.array,
        volume_second_diff: np.array
        ) -> None:
    """_summary_

    Args:
        close (np.array): _description_
        volume (np.array): _description_
    """
    # Model
    action, interpretation, color_code = solve_decision_tree(close_first_diff[-1], volume_first_diff[-1])

    # Print
    print(f"\033[9{color_code}m●\033[0m \033[1mShort term trading (noisy derivatives)\033[0m. Close {close[-1]}$ Volume {volume[-1]}\n")
    print(f"  dc/dt = {close_first_diff[-1]}")
    print(f"  dv/dt = {volume_first_diff[-1]}\n")
    print(f"  d^2c/dt^2 = {close_second_diff[-1]}")
    print(f"  d^2v/dt^2 = {volume_second_diff[-1]}\n")
    print(f"  \033[9{color_code}m{action}\033[0m ({interpretation})")
    print('')


if __name__ == "__main__":

    # Parameters
    print()
    ticker_code = input('\033[1mEnter the ticker (e.g., AAPL for Apple):\033[0m ')
    period = input('\033[1mData period to download (e.g., 30d, 3mo, 1y, max):\033[0m ')
    interval = input('\033[1mData interval (e.g., 1d, 1h, 5m):\033[0m ')
    
    # Get time series
    ticker = yf.Ticker(ticker_code)
    
    name: str = ticker.info["shortName"]
    dates, close, volume = get_time_series(ticker, period, interval)
    dates_str = [d.strftime("%d/%m/%Y") for d in dates]

    print(f"\n\033[92m●\033[0m \033[1mStock analysis\033[0m {name} ({ticker_code})")

    # Long term trading
    close_smooth = low_pass_moving_average(close, window_size=3)
    volume_smooth = low_pass_moving_average(volume, window_size=3)

    close_first_derivative: np.array = np.diff(close_smooth, axis=0)
    close_second_derivative: np.array = np.diff(close_first_derivative, axis=0)

    volume_first_derivative: np.array = np.diff(volume_smooth, axis=0)
    volume_second_derivative: np.array = np.diff(volume_first_derivative, axis=0)

    m, n = linear_regression(close_smooth)
    x = np.arange(len(close_smooth))
    y_hat = m * x + n

    # Plots
    print('')
    plt.canvas_color("black")
    plt.axes_color("black")
    plt.ticks_color("white")
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close, label="Close Price")
    plt.title(f"{name} - Close")

    plt.subplot(1, 2)
    plt.bar(dates_str, volume, label="Volume")
    plt.title(f"{name} - Volume")
    plt.show()
    print('')

    # Long term trading
    long_term_trading(m, n)

    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close_smooth, label="Close Price")
    plt.plot(dates_str, y_hat, label="Linear Regression", color="red")
    plt.title(f"{name} - Close smooth")

    plt.subplot(1, 2)
    plt.bar(dates_str, volume_smooth, label="Volume")
    plt.title(f"{name} - Volume smooth")
    plt.show()
    print('')

    # Mid term trading
    mid_term_trading(close_first_derivative, volume_first_derivative, close_second_derivative, volume_second_derivative)

    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str[1:], close_first_derivative, label="Close derivative")
    plt.title(f"{name} - Close derivative")

    plt.subplot(1, 2)
    plt.bar(dates_str[1:], volume_first_derivative, label="Volume derivative")
    plt.title(f"{name} - Volume derivative")
    plt.show()
    print('')

    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str[1:], close_second_derivative, label="Close second derivative", color="red")
    plt.title(f"{name} - Close second derivative")

    plt.subplot(1, 2)
    plt.bar(dates_str[1:], volume_second_derivative, label="Volume second derivative", color="red")
    plt.title(f"{name} - Volume second derivative")
    plt.show()
    print('')

    # Short term trading -> Considering noisy functions
    close_first_diff: np.array = np.diff(close[-3:], axis=0)
    volume_first_diff: np.array = np.diff(volume[-3:], axis=0)
    
    close_second_diff: np.array = np.diff(close_first_diff, axis=0)
    volume_second_diff: np.array = np.diff(volume_first_diff, axis=0)

    short_term_trading(close_first_diff, volume_first_diff, close_second_diff, volume_second_diff)
    