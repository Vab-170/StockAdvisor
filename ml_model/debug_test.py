import yfinance as yf
import pandas as pd

# Simple test to debug the data structure issue
df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
print("Original data columns:", df.columns.tolist())
print("Data shape:", df.shape)
print("Close type:", type(df["Close"]))
print("Close shape:", df["Close"].shape)

# Test SMA calculation
df["SMA_20"] = df["Close"].rolling(window=20).mean()
print("SMA_20 type:", type(df["SMA_20"]))
print("SMA_20 shape:", df["SMA_20"].shape)

# Test Bollinger Bands calculation
sma = df["Close"].rolling(window=20).mean()
std = df["Close"].rolling(window=20).std()
upper_band = sma + (std * 2)
lower_band = sma - (std * 2)

print("Upper band type:", type(upper_band))
print("Lower band type:", type(lower_band))
print("Upper band shape:", upper_band.shape)

# Test the problematic calculation
bb_width = (upper_band - lower_band) / df["SMA_20"]
print("BB width type:", type(bb_width))
print("BB width shape:", bb_width.shape)
