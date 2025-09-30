def print_long_term_report(dict: any) -> None:
    print(f"\n\033[4m{dict['name']}\033[0m LONG term trading (2 years)\n")
    
    print(f" - Close  = {dict['close_time_series'].values[-1]}")
    print(f" - Volume = {dict['volume_time_series'].values[-1]}\n")
    
    print(f" \033[9{dict['lr_code']}m● Linear regression\033[0m {dict['lr_description']}")
    print(f" - Slope: {dict['m']}\n")
    
    print(f" \033[9{dict['mavg_code']}m● Moving averages (SMA, EMA)\033[0m {dict['mavg_description']}")
    print(f" - SMA 100 = {dict['sma100'][-1]}  EMA 50  = {dict['ema50'][-1]}")
    print(f" - SMA 200 = {dict['sma200'][-1]}  EMA 100 = {dict['ema100'][-1]}\n")


def print_mid_term_report(dict: any) -> None:
    print(f"\n\033[4m{dict['name']}\033[0m MID term trading (18 months)\n")
    
    print(f" - Close  = {dict['close_time_series'].values[-1]}")
    print(f" - Volume = {dict['volume_time_series'].values[-1]}\n")
    
    print(f" \033[9{dict['mavg_code']}m● Moving averages (SMA, EMA)\033[0m {dict['mavg_description']}")
    print(f" - SMA 50  = {dict['sma50'][-1]}   EMA 20 = {dict['ema20'][-1]}")
    print(f" - SMA 100 = {dict['sma100'][-1]}  EMA 50 = {dict['ema50'][-1]}\n")
    
    print(f" \033[9{dict['dt_code']}m● Decision tree\033[0m {dict['dt_description']}")
    print(f" - Close derivative dc/dt  = {dict['close_time_series'].first_derivative[-1]}")
    print(f" - Volume derivative dv/dt = {dict['volume_time_series'].first_derivative[-1]}\n")
    
    print(f" \033[9{dict['macd_code']}m● MACD\033[0m {dict['macd_description']}")
    print(f" - MACD line = {dict['macd_line'][-1]}\n")
    
    print(f" \033[9{dict['rsi_code']}m● RSI\033[0m {dict['rsi_description']}")
    print(f" - RSI = {dict['rsi'][-1]}\n")