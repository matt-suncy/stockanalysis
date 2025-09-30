import numpy as np
import plotext as plt


def plot_long_term_trading(dict: any) -> None:
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