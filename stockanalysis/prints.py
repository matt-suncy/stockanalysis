import numpy as np


def print_trading_summary_long_term(
    name: str,
    close: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    ema50: np.ndarray,
    ema100: np.ndarray,
    m: float,
    n: float,
    signal: str,
    color_code: int
) -> None:
    """
    Prints a formatted long-term trading summary with linear regression and moving averages signals.

    Args:
        name (str): Ticker name or asset label.
        close (np.ndarray): Array of closing prices.
        sma100 (np.ndarray): SMA 100 values.
        sma200 (np.ndarray): SMA 200 values.
        ema50 (np.ndarray): EMA 50 values.
        ema100 (np.ndarray): EMA 100 values.
        m (float): Slope of the linear regression line.
        n (float): Intercept of the linear regression line.
        signal (str): Signal detected from moving averages.
        color_code (int): Color code for the moving averages signal.
    """
    print(f"\n\033[4m{name}\033[0m LONG term trading (2 years)\n")        
    
    print(f" - Close (now) = {close[-1]}\n")
    print(f" - Linear regression: y = mx + n")
    print(f" - m = {m}   n = {n}\n")
    print(f" - SMA 100 (now) = {sma100[-1]}  EMA 50 (now) = {ema50[-1]}")
    print(f" - SMA 200 (now) = {sma200[-1]}  EMA 100 (now) = {ema100[-1]}\n")

    print(f" \033[9{color_code}m● Moving averages (SMA, EMA) detected signal:\033[0m {signal}\n")


def print_trading_summary_mid_term(
    name: str,
    close: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    close_time_series,
    volume_time_series,
    macd_line: np.ndarray,
    rsi: np.ndarray,
    mavg_signal: str,
    mavg_color_code: int,
    action: str,
    interpretation: str,
    dt_color_code: int,
    macd_signal: str,
    macd_interpretation: str,
    macd_color_code: int,
    rsi_signal: str,
    rsi_interpretation: str,
    rsi_color_code: int
) -> None:
    """
    Prints a formatted mid-term trading summary with colored signals.

    Args:
        name (str): Ticker name or asset label.
        close (np.ndarray): Array of closing prices.
        sma50, sma100, ema20, ema50 (np.ndarray): Moving averages.
        close_time_series, volume_time_series: TimeSeries objects with first_derivative attribute.
        macd_line (np.ndarray): MACD line values.
        rsi (np.ndarray): RSI values.
        mavg_signal (str): Signal from moving averages.
        mavg_color_code (int): Color code for moving averages signal.
        action (str): Decision tree action.
        interpretation (str): Decision tree interpretation.
        dt_color_code (int): Color code for decision tree signal.
        macd_signal (str): MACD signal.
        macd_interpretation (str): MACD interpretation.
        macd_color_code (int): Color code for MACD signal.
        rsi_signal (str): RSI signal.
        rsi_interpretation (str): RSI interpretation.
        rsi_color_code (int): Color code for RSI signal.
    """
    print(f"\n\033[4m{name}\033[0m MID term trading (18 months)\n") 
      
    print(f" - Close (now) = {close[-1]}\n")
    print(f" - SMA 50 (now) = {sma50[-1]}  EMA 20 (now) = {ema20[-1]}")
    print(f" - SMA 100 (now) = {sma100[-1]}  EMA 50 (now) = {ema50[-1]}\n")
    print(f' - Close derivative dc/dt = {close_time_series.first_derivative[-1]}')
    print(f' - Volume derivative dv/dt = {volume_time_series.first_derivative[-1]}\n')
    print(f" - MACD line (now) = {macd_line[-1]}")
    print(f" - RSI (now) = {rsi[-1]}\n")
    
    print(f" \033[9{mavg_color_code}m● Moving averages (SMA, EMA) detected signal:\033[0m {mavg_signal}")
    print(f" \033[9{dt_color_code}m● Decision tree:\033[0m {action}, {interpretation}")
    print(f" \033[9{macd_color_code}m● MACD:\033[0m {macd_signal}, {macd_interpretation}")
    print(f" \033[9{rsi_color_code}m● RSI:\033[0m {rsi_signal}, {rsi_interpretation}\n")
