import numpy as np
import plotext as plt

from timeseries import *
from analysis import *
from prints import *
from plots import *


def long_term_trading(ticker: yf.Ticker, period="2y", interval="1d") -> None:
    """
    Performs a long-term trading analysis of a financial instrument using 
    historical price and volume data. 

    This function downloads historical data from Yahoo Finance, applies 
    technical indicators (SMA, EMA, linear regression), and evaluates 
    trading signals based on long-term moving average crossovers.

    Workflow:
        1. **Data acquisition**:
            - Retrieves closing prices and volume for the specified 
              ticker, period, and interval using `get_time_series`.

        2. **Time series preparation**:
            - Builds smoothed time series for closing prices (100-period) 
              and raw series for volume.

        3. **Linear regression**:
            - Fits a regression line to the smoothed closing prices:
              `y = m·x + n`
            - `m`: slope of the regression (trend direction).
            - `n`: intercept.

        4. **Technical indicators**:
            - SMA 100: Simple Moving Average over 100 periods.
            - SMA 200: Simple Moving Average over 200 periods.
            - EMA 50: Exponential Moving Average over 50 periods.
            - EMA 100: Exponential Moving Average over 100 periods.

        5. **Signal detection**:
            - Uses `signal_moving_averages_long_term` to identify 
              Golden Cross / Death Cross and price-vs-average crossovers.
            - Outputs both a textual signal and a color code.

        6. **Reporting**:
            - Prints a color-coded summary including ticker name, 
              regression equation, last close price, and current SMA/EMA values.
            - Displays the detected moving average signal.

    Args:
        ticker (yf.Ticker): Yahoo Finance ticker object representing 
            the financial instrument.
        period (str, optional): Time range for historical data. 
            Default is "2y" (2 years).
        interval (str, optional): Time step between data points. 
            Default is "1d" (daily).

    Returns:
        dict: Analysis results containing:
            - name (str): Short name of the ticker.
            - dates (np.array): Historical dates.
            - close_time_series (TimeSeries): Processed closing price series.
            - volume_time_series (TimeSeries): Processed volume series.
            - sma100 (np.array): SMA 100 values.
            - sma200 (np.array): SMA 200 values.
            - ema50 (np.array): EMA 50 values.
            - ema100 (np.array): EMA 100 values.
            - m (float): Slope of the linear regression line.
            - n (float): Intercept of the linear regression line.

    Example:
        >>> import yfinance as yf
        >>> result = long_term_trading(yf.Ticker("AAPL"))
    """
    # Get data
    name = ticker.info['shortName']
    dates, close, volume = get_time_series(ticker, period, interval)
    
    # Time series
    close_time_series = TimeSeries(close, 100)
    volume_time_series = TimeSeries(volume)
    
    # Indicators
    m, n = linear_regression(close_time_series.smooth_values)
    
    sma100 = SMA(close_time_series.values, 100)
    sma200 = SMA(close_time_series.values, 200)
    
    ema50 = EMA(close_time_series.values, 50)
    ema100 = EMA(close_time_series.values, 100)
    
    # Moving averages signal detection
    signal, color_code = signal_moving_averages_long_term(sma100, sma200, ema50, ema100, close)
    
    # Report
    print_trading_summary_long_term(
        name=name,
        close=close,
        sma100=sma100,
        sma200=sma200,
        ema50=ema50,
        ema100=ema100,
        m=m,
        n=n,
        signal=signal,
        color_code=color_code
    )
    
    # Return indicators
    return {
        "name" : name,
        "dates" : dates,
        "close_time_series" : close_time_series,
        "volume_time_series" : volume_time_series,
        "sma100" : sma100,
        "sma200" : sma200,
        "ema50" : ema50,
        "ema100": ema100,
        "m" : m,
        "n" : n
    }


def mid_term_trading(ticker: yf.Ticker, period="18mo", interval="1d") -> None:
    """
    Performs a mid-term trading analysis of a financial instrument using 
    price, volume, and multiple technical indicators. 

    This function downloads historical data from Yahoo Finance, calculates 
    moving averages, MACD, and RSI, and evaluates trading signals using 
    several complementary methods: moving average crossovers, a decision tree 
    based on derivatives, MACD interpretation, and RSI thresholds.

    Workflow:
        1. **Data acquisition**:
            - Retrieves closing prices and volume for the specified 
              ticker, period, and interval using `get_time_series`.

        2. **Time series preparation**:
            - Constructs smoothed closing price series (50-period) and 
              raw volume series.
            - Computes first derivatives (dc/dt, dv/dt) to measure trend 
              slopes and participation.

        3. **Technical indicators**:
            - SMA 50: Simple Moving Average over 50 periods (short to mid-term trend).
            - SMA 100: Simple Moving Average over 100 periods (slower mid-term trend).
            - EMA 20: Exponential Moving Average over 20 periods (fast momentum).
            - EMA 50: Exponential Moving Average over 50 periods (smoother momentum).
            - MACD line: Momentum indicator showing difference between fast 
              and slow EMAs.
            - RSI: Relative Strength Index, oscillator for overbought/oversold zones.

        4. **Signal detection**:
            - Moving Averages: Uses `signal_moving_averages_mid_term` to detect 
              crossovers and price-vs-average events.
            - Decision Tree: Uses `solve_decision_tree` to classify market 
              conditions based on dc/dt and dv/dt (price vs volume derivatives).
            - MACD: Uses `macd_signal_mid_term` to detect bullish/bearish momentum.
            - RSI: Uses `rsi_signal_mid_term` to detect overbought/oversold states.

        5. **Reporting**:
            - Prints a color-coded summary including current close price, 
              SMA/EMA values, price and volume derivatives, MACD, and RSI.
            - Displays detected signals from all methods in a consolidated 
              format.

    Args:
        ticker (yf.Ticker): Yahoo Finance ticker object representing 
            the financial instrument.
        period (str, optional): Time range for historical data. 
            Default is "18mo" (18 months).
        interval (str, optional): Time step between data points. 
            Default is "1d" (daily).

    Returns:
        dict: Analysis results containing:
            - name (str): Short name of the ticker.
            - dates (np.array): Historical dates.
            - close_time_series (TimeSeries): Processed closing price series.
            - volume_time_series (TimeSeries): Processed volume series.
            - sma50 (np.array): SMA 50 values.
            - sma100 (np.array): SMA 100 values.
            - ema20 (np.array): EMA 20 values.
            - ema50 (np.array): EMA 50 values.
            - macd_line (np.array): MACD line values.
            - rsi (np.array): RSI values.

    Example:
        >>> import yfinance as yf
        >>> result = mid_term_trading(yf.Ticker("MSFT"))
    """
    # Get data
    name = ticker.info['shortName']
    dates, close, volume = get_time_series(ticker, period, interval)
    
    # Time series
    close_time_series = TimeSeries(close, 50)
    volume_time_series = TimeSeries(volume)
    
    # Indicators
    sma50 = SMA(close_time_series.smooth_values, 50)
    sma100 = SMA(close_time_series.values, 100)
    
    ema20 = EMA(close_time_series.values, 20)
    ema50 = EMA(close_time_series.values, 50)
    
    macd_line, _, _ = MACD(close_time_series.smooth_values)
    rsi = RSI(close_time_series.smooth_values)
    
    # Signal detections
    mavg_signal, mavg_color_code = signal_moving_averages_mid_term(sma50, sma100, ema20, ema50, close)
    
    action, interpretation, dt_color_code = solve_decision_tree(
        close_time_series.first_derivative[-1],
        volume_time_series.first_derivative[-1]
    )
    
    macd_signal, macd_interpretation, macd_color_code = macd_signal_mid_term(macd_line)
    rsi_signal, rsi_interpretation, rsi_color_code = rsi_signal_mid_term(rsi)
    
    # Report
    print_trading_summary_mid_term(
        name=name,
        close=close,
        sma50=sma50,
        sma100=sma100,
        ema20=ema20,
        ema50=ema50,
        close_time_series=close_time_series,
        volume_time_series=volume_time_series,
        macd_line=macd_line,
        rsi=rsi,
        mavg_signal=mavg_signal,
        mavg_color_code=mavg_color_code,
        action=action,
        interpretation=interpretation,
        dt_color_code=dt_color_code,
        macd_signal=macd_signal,
        macd_interpretation=macd_interpretation,
        macd_color_code=macd_color_code,
        rsi_signal=rsi_signal,
        rsi_interpretation=rsi_interpretation,
        rsi_color_code=rsi_color_code
    )
    
    # Return indicators
    return {
        "name" : name,
        "dates" : dates,
        "close_time_series" : close_time_series,
        "volume_time_series" : volume_time_series,
        "sma50" : sma50,
        "sma100" : sma100,
        "ema20" : ema20,
        "ema50" : ema50,
        "macd_line" : macd_line,
        "rsi" : rsi
    }
    
    
def short_term_trading(ticker: yf.Ticker, period="2mo", interval="1d") -> None:
    """
    - Close and Volume
    - Close and Volume derivatives
    - Decision tree
    - SMA
    - SMA
    - EMA
    - EMA
    - RSI
    - ATR
    """
    pass


if __name__ == "__main__":
    
    # Plot conf
    plt.canvas_color("black")
    plt.axes_color("black")
    plt.ticks_color("white")
    
    # Parameters
    print('')
    
    ticker_code = input('\033[1mEnter the ticker (e.g., AAPL for Apple):\033[0m ')
    ticker = yf.Ticker(ticker_code)
    name: str = ticker.info["shortName"]
    
    # Analysis
    dict_long = long_term_trading(ticker)
    plot_long_term_trading(dict=dict_long)
    
    dict_mid = mid_term_trading(ticker)
    plot_mid_term_trading(dict=dict_mid)
    