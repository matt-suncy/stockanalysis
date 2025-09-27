import numpy as np


def signal_moving_averages_long_term(
    sma100: np.array, 
    sma200: np.array, 
    ema50: np.array, 
    ema100: np.array, 
    close: np.array
    ) -> tuple[str, int]:
    """
    Generates long-term trading signals based on moving average crossovers and 
    price interactions with moving averages.

    This function evaluates several moving averages and the closing price to 
    identify classic technical analysis signals such as the Golden Cross, 
    Death Cross, and buy/sell triggers when the price crosses above or below 
    specific moving averages.

    Indicators used:
        - SMA 100 (Simple Moving Average, 100 periods): 
          Measures the average price over the last 100 periods, providing a 
          medium-term trend view.
        
        - SMA 200 (Simple Moving Average, 200 periods): 
          A widely used long-term trend indicator. Often considered a 
          "line in the sand" for bull vs. bear markets.
        
        - EMA 50 (Exponential Moving Average, 50 periods): 
          A faster-moving average that reacts more quickly to recent price 
          changes, often used to detect medium-term momentum.
        
        - EMA 100 (Exponential Moving Average, 100 periods): 
          Slower than EMA 50, it balances responsiveness with stability, 
          providing confirmation of longer-term momentum.
        
        - Close price: 
          The most recent market closing price series, compared against 
          moving averages to detect crossovers.

    Signal logic:
        1. **Golden Cross (Strong Buy)**:
            Triggered when EMA 50 crosses above SMA 200, signaling potential 
            long-term bullish momentum.
        
        2. **Death Cross (Strong Sell)**:
            Triggered when EMA 50 crosses below SMA 200, signaling potential 
            long-term bearish momentum.
        
        3. **Price Crossovers with EMA/SMA**:
            - Buy signal: when the closing price crosses above a moving average.
            - Sell signal: when the closing price crosses below a moving average.
        
        Priority of signals:
            Golden Cross / Death Cross have highest priority. If no crossover 
            between EMA 50 and SMA 200 occurs, the function checks for price 
            crossing EMA 50, EMA 100, SMA 100, and SMA 200, in that order.

    Args:
        sma100 (np.array): Simple Moving Average (100 periods).
        sma200 (np.array): Simple Moving Average (200 periods).
        ema50 (np.array): Exponential Moving Average (50 periods).
        ema100 (np.array): Exponential Moving Average (100 periods).
        close (np.array): Closing price series.

    Returns:
        tuple[str, int]: 
            - signal (str): Trading signal ("Hold", "Buy", "Sell", 
              "Golden Cross (Strong Buy)", "Death Cross (Strong Sell)").
            - color_code (int): Encoded representation of the signal:
                1 = Bearish (red),
                2 = Bullish (green),
                3 = Neutral / Hold (yellow).

    Example:
        >>> signal_moving_averages_long_term(sma100, sma200, ema50, ema100, close)
        ('Golden Cross (Strong Buy)', 2)
    """
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
        
    return signal, color_code


def signal_moving_averages_mid_term(
    sma50: np.array, 
    sma100: np.array, 
    ema20: np.array, 
    ema50: np.array, 
    close: np.array
    ) -> tuple[str, int]:
    """
    Generates mid-term trading signals based on moving average crossovers and 
    price interactions with moving averages.

    This function analyzes medium-term moving averages (20 to 100 periods) and 
    the closing price to detect common technical analysis signals. It places 
    particular emphasis on the EMA 50 vs. SMA 100 crossover, often used to 
    identify changes in mid-term market trends.

    Indicators used:
        - SMA 50 (Simple Moving Average, 50 periods): 
          Captures medium-term price trends, often used by swing traders.
        
        - SMA 100 (Simple Moving Average, 100 periods): 
          Provides a slower-moving benchmark, filtering noise and 
          identifying stronger mid-term trends.
        
        - EMA 20 (Exponential Moving Average, 20 periods): 
          A fast-moving indicator that reacts quickly to price changes, 
          useful for short- to mid-term momentum detection.
        
        - EMA 50 (Exponential Moving Average, 50 periods): 
          Slower than EMA 20, it balances reactivity with stability, 
          commonly used to identify medium-term momentum.
        
        - Close price: 
          The most recent market closing price series, compared against 
          moving averages to detect crossover events.

    Signal logic:
        1. **Golden Cross (Strong Buy)**:
            Triggered when EMA 50 crosses above SMA 100, suggesting 
            strengthening bullish momentum in the mid-term.

        2. **Death Cross (Strong Sell)**:
            Triggered when EMA 50 crosses below SMA 100, suggesting 
            weakening momentum and potential bearish conditions.

        3. **Price Crossovers with EMA/SMA**:
            - Buy signal: when the closing price crosses above a moving average.
            - Sell signal: when the closing price crosses below a moving average.
        
        Priority of signals:
            Golden Cross / Death Cross have the highest weight. If no 
            EMA 50 vs. SMA 100 crossover occurs, the function evaluates 
            price crossovers in the following order:
            EMA 50 → EMA 20 → SMA 50 → SMA 100.

    Args:
        sma50 (np.array): Simple Moving Average (50 periods).
        sma100 (np.array): Simple Moving Average (100 periods).
        ema20 (np.array): Exponential Moving Average (20 periods).
        ema50 (np.array): Exponential Moving Average (50 periods).
        close (np.array): Closing price series.

    Returns:
        tuple[str, int]:
            - signal (str): Trading signal ("Hold", "Buy", "Sell", 
              "Golden Cross (Strong Buy)", "Death Cross (Strong Sell)").
            - color_code (int): Encoded representation of the signal:
                1 = Bearish (red),
                2 = Bullish (green),
                3 = Neutral / Hold (yellow).

    Example:
        >>> signal_moving_averages_mid_term(sma50, sma100, ema20, ema50, close)
        ('Buy (price crossed above EMA 20)', 2)
    """
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


def solve_decision_tree(close_derivative: float, volume_derivative: float) -> tuple[str, str, int]:
    """
    Generates a trading decision based on the directional derivatives of 
    price (close) and trading volume.

    This function applies a simple decision tree that evaluates the 
    derivatives (slopes) of the closing price and trading volume to 
    determine the strength and direction of a market trend. It outputs 
    a recommended action, an interpretation of the trend, and a color code.

    Indicators used:
        - Close derivative:
            Represents the slope of the closing price (rate of change).
            Positive values indicate upward momentum, negative values 
            indicate downward momentum.
        
        - Volume derivative:
            Represents the slope of trading volume (rate of change).
            Positive values indicate increasing participation (trend 
            confirmation), negative values indicate weakening participation 
            (trend divergence).

    Signal logic:
        1. **Price ↑, Volume ↑ (Strong Trend, Buy)**:
            - Interpretation: Price is rising and participation is 
              increasing → bullish confirmation.
            - Action: "Buy"
            - Color code: 2 (green)
        
        2. **Price ↑, Volume ↓ (Weak Trend, Hold)**:
            - Interpretation: Price is rising but participation is 
              declining → momentum may weaken.
            - Action: "Hold position"
            - Color code: 3 (yellow, neutral)
        
        3. **Price ↓, Volume ↑ (Strong Downward Trend, Sell)**:
            - Interpretation: Price is falling with increasing 
              participation → bearish confirmation.
            - Action: "Sell"
            - Color code: 1 (red)
        
        4. **Price ↓, Volume ↓ (Weak Downward Trend, Hold)**:
            - Interpretation: Price is falling but participation is 
              declining → downward momentum may weaken.
            - Action: "Hold position"
            - Color code: 3 (yellow, neutral)

    Args:
        close_derivative (float): Derivative of closing price (trend slope).
        volume_derivative (float): Derivative of trading volume (participation slope).

    Returns:
        tuple[str, str, int]:
            - action (str): Suggested trading action ("Buy", "Sell", "Hold position").
            - interpretation (str): Explanation of trend strength and direction.
            - color_code (int): Encoded representation of the signal:
                1 = Bearish (red),
                2 = Bullish (green),
                3 = Neutral / Hold (yellow).

    Example:
        >>> solve_decision_tree(0.05, 0.10)
        ('Buy', 'strong trend', 2)
    """
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


def macd_signal_mid_term(macd_line: np.array) -> tuple[str, str, int]:
    """
    Interprets the MACD line to generate trading signals based on 
    momentum shifts and trend strength.

    The Moving Average Convergence Divergence (MACD) is a momentum indicator 
    that shows the relationship between two moving averages of price. 
    This function focuses only on the MACD line relative to the zero level 
    (centerline) and its recent trend direction.

    Indicators used:
        - MACD line:
            Computed as the difference between a short-term EMA and a 
            long-term EMA (commonly EMA 12 and EMA 26). 
            Positive values indicate bullish momentum, while negative 
            values indicate bearish momentum.
    
    Signal logic:
        1. **Bullish signals**:
            - Zero crossover:
                If MACD crosses from below zero to above zero, momentum 
                shifts bullish.
                → Action: "Buy"
                → Interpretation: "MACD crossed above zero → bullish momentum"
                → Color: 2 (green)
            - Positive and rising:
                If MACD is above zero and continues to rise, it suggests 
                strengthening bullish trend.
                → Action: "Buy"
                → Interpretation: "MACD positive and rising → bullish trend"
                → Color: 2 (green)
        
        2. **Bearish signals**:
            - Zero crossover:
                If MACD crosses from above zero to below zero, momentum 
                shifts bearish.
                → Action: "Sell"
                → Interpretation: "MACD crossed below zero → bearish momentum"
                → Color: 1 (red)
            - Negative and falling:
                If MACD is below zero and continues to fall, it suggests 
                strengthening bearish trend.
                → Action: "Sell"
                → Interpretation: "MACD negative and falling → bearish trend"
                → Color: 1 (red)

        3. **Default case**:
            If no bullish or bearish condition is met, 
            → Action: "Hold"
            → Interpretation: "Neutral momentum"
            → Color: 3 (yellow, neutral)

    Args:
        macd_line (np.array): Array of MACD line values.

    Returns:
        tuple[str, str, int]:
            - macd_signal (str): Suggested trading action ("Buy", "Sell", "Hold").
            - macd_interpretation (str): Explanation of the MACD behavior.
            - color_code (int): Encoded representation of the signal:
                1 = Bearish (red),
                2 = Bullish (green),
                3 = Neutral / Hold (yellow).

    Example:
        >>> solve_macd(np.array([-0.2, 0.1]))
        ('Buy', 'MACD crossed above zero → bullish momentum', 2)
    """
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


def rsi_signal_mid_term(rsi: np.array) -> tuple[str, str, int]:
    """
    Interprets the Relative Strength Index (RSI) to generate trading signals 
    based on overbought and oversold conditions.

    The RSI is a momentum oscillator that measures the speed and magnitude 
    of recent price changes, oscillating between 0 and 100. It is often used 
    to identify potential reversal points when the market becomes overbought 
    or oversold.

    Indicators used:
        - RSI (Relative Strength Index):
            A momentum indicator typically calculated over 14 periods.
            - RSI > 70: market may be overbought (potential sell zone).
            - RSI < 30: market may be oversold (potential buy zone).
            - RSI between 30–70: neutral, trend continuation likely.

    Signal logic:
        1. **Overbought (Sell signal)**:
            - If RSI > 70, the market is considered overbought.
            - Interpretation: Possible price correction or reversal downward.
            - Action: "Sell"
            - Color: 1 (red)

        2. **Oversold (Buy signal)**:
            - If RSI < 30, the market is considered oversold.
            - Interpretation: Possible price rebound or reversal upward.
            - Action: "Buy"
            - Color: 2 (green)

        3. **Neutral (Hold)**:
            - If 30 ≤ RSI ≤ 70, momentum is neutral.
            - Interpretation: "RSI neutral"
            - Action: "Hold"
            - Color: 3 (yellow, neutral)

    Args:
        rsi (np.array): Array of RSI values (0–100 scale).

    Returns:
        tuple[str, str, int]:
            - rsi_signal (str): Suggested trading action ("Buy", "Sell", "Hold").
            - rsi_interpretation (str): Explanation of RSI condition.
            - color_code (int): Encoded representation of the signal:
                1 = Bearish (red),
                2 = Bullish (green),
                3 = Neutral / Hold (yellow).

    Example:
        >>> rsi_signal(np.array([45, 72]))
        ('Sell', 'RSI > 70 → Overbought, possible sell signal', 1)
    """
    rsi_signal = "Hold"
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