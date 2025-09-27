import numpy as np
import plotext as plt


def plot_long_term_trading(dict: any) -> None:
    """
    Plots long-term trading indicators including linear regression, SMA, EMA, 
    and trading volume using matplotlib.

    This function generates two sets of subplots:
    1. Close price with linear regression and trading volume.
    2. Close price with SMA (100, 200) and EMA (50, 100).

    Args:
        dict (any): Dictionary containing the following keys:
            - 'dates' (list[datetime]): List of dates corresponding to the time series.
            - 'name' (str): Name of the stock or asset.
            - 'close_time_series' (object): Object with attributes:
                - values (list[float]): Close price values.
                - smooth_values (list[float]): Smoothed close price values used in regression.
            - 'volume_time_series' (object): Object with attribute:
                - values (list[float]): Volume values.
            - 'm' (float): Slope of the linear regression line.
            - 'n' (float): Intercept of the linear regression line.
            - 'sma100' (list[float]): Simple Moving Average over 100 periods.
            - 'sma200' (list[float]): Simple Moving Average over 200 periods.
            - 'ema50' (list[float]): Exponential Moving Average over 50 periods.
            - 'ema100' (list[float]): Exponential Moving Average over 100 periods.

    Returns:
        None: The function displays matplotlib plots and prints a line break.
    """
    dates_str = [d.strftime("%d/%m/%Y") for d in dict['dates']]
    
    x = np.arange(len(dict['close_time_series'].smooth_values))
    y_hat = dict['m'] * x + dict['n']
    
    # Plot close and volume
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, dict['close_time_series'].values, label="Close", color="blue")
    plt.plot(dates_str, y_hat, label="Linear Regression", color="red")
    plt.title(f"{dict['name']} - Close")

    plt.subplot(1, 2)
    plt.bar(dates_str, dict['volume_time_series'].values, label="Volume")
    plt.title(f"{dict['name']} - Volume")
    plt.show()
    print('')
    
    # Plots SMA y EMA
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, dict['close_time_series'].values, label="Close", color="blue")
    plt.plot(dates_str, dict['sma100'], label="SMA 100", color="red")
    plt.plot(dates_str, dict['sma200'], label="SMA 200", color="green")
    plt.title(f"{dict['name']} - SMA")

    plt.subplot(1, 2)
    plt.plot(dates_str, dict['close_time_series'].values, label="Close", color="blue")
    plt.plot(dates_str, dict['ema50'], label="EMA 50", color="red")
    plt.plot(dates_str, dict['ema100'], label="EMA 100", color="green")
    plt.title(f"{dict['name']} - EMA")
    
    plt.show()
    print('')
    

def plot_mid_term_trading(dict: any) -> None:
    """
    Plots mid-term trading indicators including close price, trading volume,
    derivatives, SMA, EMA, MACD, and RSI using matplotlib.

    This function generates four sets of subplots:
    1. Close price and trading volume.
    2. First derivatives of close price and volume.
    3. Close price with SMA (50, 100) and EMA (20, 50).
    4. MACD and RSI indicators.

    Args:
        dict (any): Dictionary containing the following keys:
            - 'dates' (list[datetime]): List of dates corresponding to the time series.
            - 'name' (str): Name of the stock or asset.
            - 'close_time_series' (object): Object with attributes:
                - values (list[float]): Close price values.
                - first_derivative (list[float]): First derivative of close prices.
            - 'volume_time_series' (object): Object with attributes:
                - values (list[float]): Volume values.
                - first_derivative (list[float]): First derivative of volume values.
            - 'sma50' (list[float]): Simple Moving Average over 50 periods.
            - 'sma100' (list[float]): Simple Moving Average over 100 periods.
            - 'ema20' (list[float]): Exponential Moving Average over 20 periods.
            - 'ema50' (list[float]): Exponential Moving Average over 50 periods.
            - 'macd_line' (list[float]): MACD line values.
            - 'rsi' (list[float]): Relative Strength Index values.

    Returns:
        None: The function displays matplotlib plots and prints line breaks.
    """
    dates_str = [d.strftime("%d/%m/%Y") for d in dict['dates']]
    
    # Plot close and volume
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, dict['close_time_series'].values, label="Close", color="blue")
    plt.title(f"{dict['name']} - Close")

    plt.subplot(1, 2)
    plt.bar(dates_str, dict['volume_time_series'].values, label="Volume")
    plt.title(f"{dict['name']} - Volume")
    plt.show()
    print('')
    
    # Plot close and volume derivatives
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, dict['close_time_series'].first_derivative, label="Close derivative", color="red")
    plt.title(f"{dict['name']} - Close derivative")

    plt.subplot(1, 2)
    plt.bar(dates_str, dict['volume_time_series'].first_derivative, label="Volume derivative", color="red")
    plt.title(f"{dict['name']} - Volume derivative")
    plt.show()
    print('')
    
    # Plots SMA y EMA
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, dict['close_time_series'].values, label="Close", color="blue")
    plt.plot(dates_str, dict['sma50'], label="SMA 50", color="red")
    plt.plot(dates_str, dict['sma100'], label="SMA 100", color="green")
    plt.title(f"{dict['name']} - SMA")

    plt.subplot(1, 2)
    plt.plot(dates_str, dict['close_time_series'].values, label="Close", color="blue")
    plt.plot(dates_str, dict['ema20'], label="EMA 20", color="red")
    plt.plot(dates_str, dict['ema50'], label="EMA 50", color="green")
    plt.title(f"{dict['name']} - EMA")
    
    plt.show()
    print('')
    
    # Plots MACD y RSI
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, dict['macd_line'], label="MACD", color="green")
    plt.title(f"{dict['name']} - MACD")

    plt.subplot(1, 2)
    plt.plot(dates_str, dict['rsi'], label="RSI", color="green")
    plt.title(f"{dict['name']} - RSI")
    
    plt.show()
    print('')