import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta




print("Fetching latest stock data...")
df = pd.read_csv("S&P 500 Project - Sheet1 copy.csv")
tickers = df["Symbol"].tolist()
new_data = yf.download(tickers, period="2y", auto_adjust=True)["Close"]
new_data.to_csv("s&p.csv")
print("Data updated!")

data = pd.read_csv("s&p.csv", index_col=0, parse_dates=True)

def lstm_predict(ticker, days_forward=90):
    if ticker not in data.columns:
        print(f"{ticker} not found")
        return

    prices = data[ticker].dropna().values.reshape(-1, 1)

    if len(prices) < 120:
        print(f"{ticker} has insufficient data")
        return

    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    
    X, y = [], []
    for i in range(120, len(scaled)):
        X.append(scaled[i-120:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(120, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    print(f"Training LSTM for {ticker}...")
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    
    last_120 = scaled[-120:]
    predictions = []
    current_sequence = last_120.copy()

    for _ in range(days_forward):
        input_seq = current_sequence[-120:].reshape(1, 120, 1)
        next_price = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_price)
        current_sequence = np.append(current_sequence, [[next_price]], axis=0)

    
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

 
    last_date = data[ticker].dropna().index[-1]
    future_dates = []
    current_date = last_date
    while len(future_dates) < days_forward:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            future_dates.append(current_date)

    future_series = pd.Series(predictions, index=future_dates)
    historical = data[ticker].dropna()

    
    plt.figure(figsize=(14, 6))
    plt.plot(historical, label="Historical", color="blue")
    plt.plot(future_series, label=f"LSTM Forecast ({days_forward} days)", 
             color="red", linestyle="--")
    plt.axvline(x=last_date, color="gray", linestyle=":", label="Today")
    plt.title(f"{ticker} - LSTM Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    current_price = historical.iloc[-1]
    final_price = predictions[-1]
    pct_change = ((final_price - current_price) / current_price) * 100
    print(f"\n{ticker} LSTM Forecast Summary:")
    print(f"Current Price:  ${current_price:.2f}")
    print(f"Predicted Price in {days_forward} trading days: ${final_price:.2f}")
    print(f"Predicted Change: {pct_change:.2f}%")

lstm_predict("AAPL")