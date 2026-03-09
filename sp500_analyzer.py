import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("s&p.csv", index_col = 0, parse_dates=True)

def analyze(ticker):
    if ticker not in df.columns:
        print("Stock is unable to be found")
        return
    
    prices = df[ticker].dropna()


    midpoint = round(len(prices)/2)

    historical = prices[:midpoint]

    actual = prices[midpoint:]
    avg_daily_return = historical.pct_change().mean()
    projected = []
    last_price = historical.iloc[-1]
    for i in range(len(actual)):
        last_price = last_price * (1 + avg_daily_return)
        projected.append(last_price)
    projected = pd.Series(projected, index=actual.index)
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(historical, label="Historical", color="blue")
    plt.plot(actual, label="Actual", color="green")
    plt.plot(projected, label="Projected", color="red", linestyle="--")
    plt.title(f"{ticker} - Projected vs Actual Performance")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()


analyze("AAPL")