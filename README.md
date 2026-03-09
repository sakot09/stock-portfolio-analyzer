S&P 500 LSTM Stock Price Forecaster
A Python tool that trains a Long Short-Term Memory (LSTM) neural network on historical price data for all 503 S&P 500 companies and generates 90-day forward price forecasts.
What It Does

Automatically downloads up-to-date historical price data for all S&P 500 companies
Trains an LSTM neural network on 5 years of price history
Predicts the next 90 trading days of price movement for any stock
Generates a chart showing historical prices and the forward forecast
Prints a summary showing current price, predicted price, and projected % change

How It Works
The model looks at 120 trading days of price history at a time and learns patterns in how prices move. It uses recursive forecasting (each predicted day feeds into the next prediction) to generate a full 90-day outlook. The model is retrained on the latest data every time it runs, so forecasts are always current.

Tech Stack

Python
TensorFlow / Keras — LSTM neural network
yfinance — real-time stock data
pandas — data manipulation
NumPy — numerical computing
scikit-learn — data preprocessing
Matplotlib — data visualization

Setup
1. Clone the repository
git clone https://github.com/sanay-kotian/stock-portfolio-analyzer
cd stock-portfolio-analyzer
2. Install dependencies
pip3 install yfinance pandas matplotlib tensorflow scikit-learn
3. Add S&P 500 tickers
Download the S&P 500 company list from Wikipedia and save it as sp500.csv in the project folder with a column named Symbol.
4. Run the forecaster
python3 lstm_analyzer.py
Usage
To forecast any S&P 500 stock, change the ticker at the bottom of lstm_analyzer.py:
pythonlstm_predict("AAPL")   # Apple
lstm_predict("TSLA")   # Tesla
lstm_predict("JPM")    # JPMorgan Chase
lstm_predict("GOOGL")  # Google
To change the forecast horizon:
pythonlstm_predict("AAPL", days_forward=180)  # 180 day forecast
Example Output
Fetching latest stock data...
Data updated!
Training LSTM for AAPL...
Epoch 1/50 - loss: 0.0023
...
Epoch 50/50 - loss: 0.0004

AAPL LSTM Forecast Summary:
Current Price:  $213.49
Predicted Price in 90 trading days: $228.17
Predicted Change: +6.88%
Limitations

Stock markets are inherently unpredictable.This model is for educational purposes and should not be used as financial advice
LSTM forecasts based purely on price history do not account for news, earnings, or macroeconomic events
Recursive forecasting compounds prediction error over time. Shorter forecasts are more reliable than longer ones

What I Learned

How to build and train LSTM neural networks using TensorFlow/Keras
Time series forecasting and recursive prediction techniques
Data preprocessing including MinMax scaling and sequence generation
Working with real financial data at scale (500+ companies, 5 years of history)
Data visualization with Matplotlib

Author
Sanay Kotian

sanaykotian0@gmail.com