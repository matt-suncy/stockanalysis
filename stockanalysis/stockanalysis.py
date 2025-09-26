import numpy as np
import plotext as plt

from timeseries import *


SLOPE_THRESHOLD: float = 0.01


class TimeSeries:
    
    def __init__(self, values: np.array, window_size=3) -> None:
        self.values = values
        self.smooth_values = low_pass(values, window_size)
        self.first_derivative = np.gradient(self.smooth_values)
        self.second_derivative = np.gradient(self.first_derivative)


def long_term_trading(ticker: yf.Ticker, period="2y", interval="1d") -> None:
    """Long-term market analysis using price and volume time series with
    moving averages and linear regression.

    Overview:
    ----------
    The goal of this function is to analyze the long-term trend of a stock
    by combining smoothed data, linear regression, and moving averages.

    Indicators used:
    ----------------
    - Linear Regression of smoothed prices:
        Provides the general slope (trend) of the stock, ignoring short-term noise.
    
    - SMA 100:
        Very long-term trend indicator. Smooth but reacts slower to price changes.

    - SMA 200:
        The most important long-term indicator.
        Represents the fundamental price trend (approx. 1 year of trading).
    
    - EMA 50:
        Shorter-term but still long horizon (approx. 2.5 months).
        Reacts faster than SMA 200 and captures primary market movements.

    - EMA 100:
        A compromise between EMA 50’s responsiveness and SMA 200’s stability.

    Trading signals:
    ----------------
    - Bullish trend:
        Price > moving averages → Moving average acts as dynamic support.
        Buy signal if price crosses above a moving average.

    - Bearish trend:
        Price < moving averages → Moving average acts as dynamic resistance.
        Sell signal if price crosses below a moving average.

    - Moving Average Crossovers:
        These provide stronger trading signals:
        
        * Golden Cross:
            Short-term EMA crosses above long-term SMA.
            This indicates a strong buy signal (bullish reversal).
        
        * Death Cross:
            Short-term EMA crosses below long-term SMA.
            This indicates a strong sell signal (bearish reversal).

    Args:
        ticker (yf.Ticker): Stock ticker object from yfinance.
        period (str, optional): Time period for historical data. Default = "2y".
        interval (str, optional): Data interval. Default = "1d".
    """
    # Get data
    dates, close, volume = get_time_series(ticker, period, interval)
    dates_str = [d.strftime("%d/%m/%Y") for d in dates]
    
    # Time series
    close_time_series = TimeSeries(close, 100)
    volume_time_series = TimeSeries(volume)
    
    # Linear regression
    m, n = linear_regression(close_time_series.smooth_values)

    x = np.arange(len(close_time_series.smooth_values))
    y_hat = m * x + n
    
    # SMA, EMA
    sma100 = SMA(close_time_series.values, 100)
    sma200 = SMA(close_time_series.values, 200)
    
    ema50 = EMA(close_time_series.values, 50)
    ema100 = EMA(close_time_series.values, 100)
    
    # Signal detection
    signal = "Hold"  # default
    color_code = 3   # default
    
    # Golden Cross / Death Cross
    if ema50[-2] < sma200[-2] and ema50[-1] > sma200[-1]:
        signal = "Golden Cross (Strong Buy)"
        color_code = 2
    elif ema50[-2] > sma200[-2] and ema50[-1] < sma200[-1]:
        signal = "Death Cross (Strong Sell)"
        color_code = 1
    
    # Buy / Sell signals based on price crossing moving averages
    elif close[-2] < ema50[-2] and close[-1] > ema50[-1]:
        signal = "Buy (price crossed above EMA 50)"
        color_code = 2
    elif close[-2] > ema50[-2] and close[-1] < ema50[-1]:
        signal = "Sell (price crossed below EMA 50)"
        color_code = 1

    elif close[-2] < ema100[-2] and close[-1] > ema100[-1]:
        signal = "Buy (price crossed above EMA 100)"
        color_code = 2
    elif close[-2] > ema100[-2] and close[-1] < ema100[-1]:
        signal = "Sell (price crossed below EMA 100)"
        color_code = 1

    elif close[-2] < sma100[-2] and close[-1] > sma100[-1]:
        signal = "Buy (price crossed above SMA 100)"
        color_code = 2
    elif close[-2] > sma100[-2] and close[-1] < sma100[-1]:
        signal = "Sell (price crossed below SMA 100)"
        color_code = 1

    elif close[-2] < sma200[-2] and close[-1] > sma200[-1]:
        signal = "Buy (price crossed above SMA 200)"
        color_code = 2
    elif close[-2] > sma200[-2] and close[-1] < sma200[-1]:
        signal = "Sell (price crossed below SMA 200)"
        color_code = 1
    
    name = ticker.info['shortName']
    print(f"\n\033[9{color_code}m●\033[0m \033[1m{name}\033[0m LONG term trading (2 years)\n")        
    print(f"Linear regression: y = mx + n")
    print(f"m = {m}")
    print(f"n = {n}\n")
    # Print report
    print(f"Close (now) = {close[-1]}\n")
    print(f"SMA 100 (now) = {sma100[-1]}  EMA 50 (now) = {ema50[-1]}")
    print(f"SMA 200 (now) = {sma200[-1]}  EMA 100 (now) = {ema100[-1]}\n")

    print(f"\033[9{color_code}mDetected Signal:\033[0m {signal}\n")
    
    # Plot close and volume
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close_time_series.values, label="Close", color="blue")
    plt.plot(dates_str, y_hat, label="Linear Regression", color="red")
    plt.title(f"{name} - Close")

    plt.subplot(1, 2)
    plt.bar(dates_str, volume_time_series.values, label="Volume")
    plt.title(f"{name} - Volume")
    plt.show()
    print('')
    
    # Plots SMA y EMA
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close_time_series.values, label="Close", color="blue")
    plt.plot(dates_str, sma100, label="SMA 100", color="red")
    plt.plot(dates_str, sma200, label="SMA 200", color="green")
    plt.title(f"{name} - SMA")

    plt.subplot(1, 2)
    plt.plot(dates_str, close_time_series.values, label="Close", color="blue")
    plt.plot(dates_str, ema50, label="EMA 50", color="red")
    plt.plot(dates_str, ema100, label="EMA 100", color="green")
    plt.title(f"{name} - EMA")
    
    plt.show()
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

def solve_sma_ema(
    sma50: np.array, 
    sma100: np.array, 
    ema20: np.array, 
    ema50: np.array, 
    close: np.array
    ) -> tuple[str, int]:
    signal = 'Hold'
    color_code = 3

    # Golden / Death Cross (EMA50 vs SMA100)
    if ema50[-2] < sma100[-2] and ema50[-1] > sma100[-1]:
        signal = "Golden Cross (Strong Buy)"
        color_code = 2
    elif ema50[-2] > sma100[-2] and ema50[-1] < sma100[-1]:
        signal = "Death Cross (Strong Sell)"
        color_code = 1

    # Buy / Sell signals based on price crossing EMA50
    elif close[-2] < ema50[-2] and close[-1] > ema50[-1]:
        signal = "Buy (price crossed above EMA 50)"
        color_code = 2
    elif close[-2] > ema50[-2] and close[-1] < ema50[-1]:
        signal = "Sell (price crossed below EMA 50)"
        color_code = 1

    # Buy / Sell signals based on price crossing EMA20
    elif close[-2] < ema20[-2] and close[-1] > ema20[-1]:
        signal = "Buy (price crossed above EMA 20)"
        color_code = 2
    elif close[-2] > ema20[-2] and close[-1] < ema20[-1]:
        signal = "Sell (price crossed below EMA 20)"
        color_code = 1

    # Buy / Sell signals based on price crossing SMA50
    elif close[-2] < sma50[-2] and close[-1] > sma50[-1]:
        signal = "Buy (price crossed above SMA 50)"
        color_code = 2
    elif close[-2] > sma50[-2] and close[-1] < sma50[-1]:
        signal = "Sell (price crossed below SMA 50)"
        color_code = 1

    # Buy / Sell signals based on price crossing SMA100
    elif close[-2] < sma100[-2] and close[-1] > sma100[-1]:
        signal = "Buy (price crossed above SMA 100)"
        color_code = 2
    elif close[-2] > sma100[-2] and close[-1] < sma100[-1]:
        signal = "Sell (price crossed below SMA 100)"
        color_code = 1
        
    return signal, color_code

def solve_macd(macd_line: np.array):
    # Interpret MACD
    macd_signal = 'Hold'  # Default
    macd_interpretation = 'Neutral momentum'
    color_code = 3

    # Bullish signals    
    if macd_line[-2] < 0 and macd_line[-1] > 0:
        macd_signal = "Buy"
        macd_interpretation = "MACD crossed above zero → bullish momentum"
        color_code = 2
    elif macd_line[-1] > 0 and macd_line[-1] > macd_line[-2]:
        macd_signal = "Buy"
        macd_interpretation = "MACD positive and rising → bullish trend"
        color_code = 2

    # Bearish signals
    elif macd_line[-2] > 0 and macd_line[-1] < 0:
        macd_signal = "Sell"
        macd_interpretation = "MACD crossed below zero → bearish momentum"
        color_code = 1
    elif macd_line[-1] < 0 and macd_line[-1] < macd_line[-2]:
        macd_signal = "Sell"
        macd_interpretation = "MACD negative and falling → bearish trend"
        color_code = 1
        
    return macd_signal, macd_interpretation, color_code


def solve_rsi(rsi: np.array) -> tuple[str, str, int]:
    rsi_signal = "Hold"  # por defecto
    rsi_interpretation = "RSI neutral"
    color_code = 3

    if rsi[-1] > 70:
        rsi_signal = "Sell"
        rsi_interpretation = "RSI > 70 → Overbought, possible sell signal"
        color_code = 1
    elif rsi[-1] < 30:
        rsi_signal = "Buy"
        rsi_interpretation = "RSI < 30 → Oversold, possible buy signal"
        color_code = 2
        
    return rsi_signal, rsi_interpretation, color_code


def mid_term_trading(ticker: yf.Ticker, period="18mo", interval="1d") -> None:
    """
    Perform a comprehensive mid-term trading analysis for a given stock ticker using 
    multiple technical indicators and decision-making heuristics over approximately 18 months.

    The function performs the following steps and provides trading signals based on them:

    1. Fetch historical stock data:
       - Close prices: Used for trend and momentum analysis.
       - Trading volume: Helps confirm the strength of price movements.

    2. Compute smoothed time series and derivatives:
       - Smooth the close prices to reduce noise and calculate the first derivative.
       - Close derivative indicates the rate of price change:
         - Positive → price is rising (uptrend)
         - Negative → price is falling (downtrend)
       - Volume derivative indicates the change in trading activity:
         - Positive → increasing interest
         - Negative → decreasing interest

    3. Decision Tree based on derivatives:
       - Strong trend (Buy) if both close and volume derivatives are positive → bullish momentum
       - Weak trend (Hold) if close is rising but volume is decreasing → trend may lack strength
       - Strong downward trend (Sell) if close is falling and volume is increasing → strong bearish momentum
       - Weak downward trend (Hold) if both close and volume derivatives are decreasing → mild downtrend

    4. Calculate moving averages (SMA and EMA):
       - SMA50 (50-day Simple Moving Average): Reflects medium-term trend
       - SMA100 (100-day Simple Moving Average): Reflects long-term trend
       - EMA20 (20-day Exponential Moving Average): More sensitive to recent price changes
       - EMA50 (50-day Exponential Moving Average): Balances medium-term and recent price trends

    5. Generate trading signals based on moving averages:
       - Golden Cross (EMA50 crosses above SMA100): Strong Buy signal, bullish long-term trend
       - Death Cross (EMA50 crosses below SMA100): Strong Sell signal, bearish long-term trend
       - Price crossing EMA20, EMA50, SMA50, SMA100:
         - Price crosses above → Buy signal (potential upward movement)
         - Price crosses below → Sell signal (potential downward movement)

    6. Calculate MACD (Moving Average Convergence Divergence):
       - MACD is the difference between fast and slow EMAs
       - Signals:
         - MACD crosses above zero → bullish momentum (Buy)
         - MACD positive and rising → bullish trend continuation (Buy)
         - MACD crosses below zero → bearish momentum (Sell)
         - MACD negative and falling → bearish trend continuation (Sell)

    7. Calculate RSI (Relative Strength Index):
       - RSI measures overbought or oversold conditions (0-100 scale)
       - Signals:
         - RSI > 70 → Overbought, possible Sell signal
         - RSI < 30 → Oversold, possible Buy signal
         - RSI between 30-70 → Neutral, Hold position

    8. Print summary of current prices, indicator values, and trading signals:
       - Displays SMA, EMA, MACD, RSI, derivatives, and decision tree results
       - Color codes used for signals:
         - 1 = Bearish / Sell
         - 2 = Bullish / Buy
         - 3 = Neutral / Hold

    9. Plot visualizations:
       - Close prices and trading volume
       - Derivatives of close prices and volume
       - SMA and EMA overlays for trend visualization

    Args:
        ticker (yf.Ticker): The Yahoo Finance Ticker object to analyze.
        period (str, optional): Look-back period for historical data (default: "18mo").
        interval (str, optional): Data interval (default: "1d").

    Returns:
        None. Prints trading signals and plots visualizations.
    """
    # Get data
    dates, close, volume = get_time_series(ticker, period, interval)
    dates_str = [d.strftime("%d/%m/%Y") for d in dates]
    
    # Time series
    close_time_series = TimeSeries(close, 50)
    volume_time_series = TimeSeries(volume)
    
    # SMA, EMA
    sma50 = SMA(close_time_series.smooth_values, 50)
    sma100 = SMA(close_time_series.values, 100)
    
    ema20 = EMA(close_time_series.values, 20)
    ema50 = EMA(close_time_series.values, 50)
    
    # MACD
    macd_line, _, _ = MACD(close_time_series.smooth_values)
    
    # RSI
    rsi = RSI(close_time_series.smooth_values)
    
    # MAVG actions Buy / Sell signals based on price crossing moving averages
    mavg_signal, mavg_color_code = solve_sma_ema(sma50, sma100, ema20, ema50, close)
    
    # Decision tree
    action, interpretation, dt_color_code = solve_decision_tree(
        close_time_series.first_derivative[-1],
        volume_time_series.first_derivative[-1]
    )
    
    # MACD actions
    macd_signal, macd_interpretation, macd_color_code = solve_macd(macd_line)
    
    # RSI actions
    rsi_signal, rsi_interpretation, rsi_color_code = solve_rsi(rsi)

    name = ticker.info['shortName']
    print(f"\n\033[9{mavg_color_code}m●\033[0m \033[1m{name}\033[0m MID term trading (18 months)\n")        
    print(f"Close (now) = {close[-1]}\n")
    print(f"SMA 50 (now) = {sma50[-1]}  EMA 20 (now) = {ema20[-1]}")
    print(f"SMA 100 (now) = {sma100[-1]}  EMA 50 (now) = {ema50[-1]}\n")
    print(f'Close derivative dc/dt = {close_time_series.first_derivative[-1]}')
    print(f'Volume derivative dv/dt = {volume_time_series.first_derivative[-1]}\n')
    print(f"MACD line (now) = {macd_line[-1]}")
    print(f"RSI (now) = {rsi[-1]}\n")
    
    print(f"\033[9{mavg_color_code}mDetected Signal:\033[0m {mavg_signal}")
    print(f"\033[9{dt_color_code}mDecision tree:\033[0m {action}, {interpretation}")
    print(f"\033[9{macd_color_code}mMACD:\033[0m {macd_signal}, {macd_interpretation}")
    print(f"\033[9{rsi_color_code}mRSI:\033[0m {rsi_signal}, {rsi_interpretation}\n")
    
    # Plot close and volume
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close_time_series.values, label="Close", color="blue")
    plt.title(f"{name} - Close")

    plt.subplot(1, 2)
    plt.bar(dates_str, volume_time_series.values, label="Volume")
    plt.title(f"{name} - Volume")
    plt.show()
    print('')
    
    # Plot close and volume derivatives
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close_time_series.first_derivative, label="Close derivative", color="red")
    plt.title(f"{name} - Close derivative")

    plt.subplot(1, 2)
    plt.bar(dates_str, volume_time_series.first_derivative, label="Volume derivative", color="red")
    plt.title(f"{name} - Volume derivative")
    plt.show()
    print('')
    
    # Plots SMA y EMA
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, close_time_series.values, label="Close", color="blue")
    plt.plot(dates_str, sma50, label="SMA 50", color="red")
    plt.plot(dates_str, sma100, label="SMA 100", color="green")
    plt.title(f"{name} - SMA")

    plt.subplot(1, 2)
    plt.plot(dates_str, close_time_series.values, label="Close", color="blue")
    plt.plot(dates_str, ema20, label="EMA 20", color="red")
    plt.plot(dates_str, ema50, label="EMA 50", color="green")
    plt.title(f"{name} - EMA")
    
    plt.show()
    print('')
    
    # Plots MACD y RSI
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.plot(dates_str, macd_line, label="MACD", color="green")
    plt.title(f"{name} - MACD")

    plt.subplot(1, 2)
    plt.plot(dates_str, rsi, label="RSI", color="green")
    plt.title(f"{name} - RSI")
    
    plt.show()
    print('')
    
    
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
    
    plt.canvas_color("black")
    plt.axes_color("black")
    plt.ticks_color("white")
    
    # Parameters
    print()
    
    ticker_code = input('\033[1mEnter the ticker (e.g., AAPL for Apple):\033[0m ')
    ticker = yf.Ticker(ticker_code)
    name: str = ticker.info["shortName"]
    
    long_term_trading(ticker)
    mid_term_trading(ticker)
    