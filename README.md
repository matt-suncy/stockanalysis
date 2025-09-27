# stockanalysis
This project provides a Python-based toolkit for **financial market analysis** using data from Yahoo Finance.  
It helps investors and traders make more informed decisions by applying **technical indicators**, **trend analysis**, and **signal detection** on historical price and volume data.  

![](img/1.png)

![](img/2.png)

## Features  
- **Long-term analysis (2 years, daily data)**  
  - Moving Averages (SMA 100, SMA 200, EMA 50, EMA 100)  
  - Linear regression trend fitting  
  - Golden Cross / Death Cross signal detection  

- **Mid-term analysis (18 months, daily data)**  
  - Moving Averages (SMA 50, SMA 100, EMA 20, EMA 50)  
  - Price and volume derivatives (momentum and participation)  
  - MACD (bullish/bearish momentum detection)  
  - RSI (overbought/oversold states)  
  - Decision tree combining price & volume signals  

- **Short-term analysis**
  - TODO

## Dependencies
numpy
yfinance
plotext

## Usage  
Run the script and enter the stock ticker symbol (e.g., `AAPL` for Apple, `MSFT` for Microsoft)