import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import datetime

# Utility functions for technical indicators
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    """Compute MACD (Moving Average Convergence Divergence)"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(series, period=20, std_dev=2):
    """Compute Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (series - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bb_position

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """Compute Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_atr(high, low, close, period=14):
    """Compute Average True Range (ATR) for volatility"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def get_market_context():
    """Get market-wide indicators for context"""
    try:
        # VIX (Fear Index)
        vix = yf.download("^VIX", period="30d", progress=False)
        vix_current = float(vix["Close"].iloc[-1])
        
        # 10-Year Treasury Yield
        treasury = yf.download("^TNX", period="30d", progress=False)
        treasury_yield = float(treasury["Close"].iloc[-1])
        
        # Dollar Index
        dxy = yf.download("DX-Y.NYB", period="30d", progress=False)
        dollar_index = float(dxy["Close"].iloc[-1])
        
        # S&P 500 momentum
        spy = yf.download("SPY", period="30d", progress=False)
        spy_momentum = float(spy["Close"].pct_change(periods=5).iloc[-1])
        
        # Tech sector momentum
        qqq = yf.download("QQQ", period="30d", progress=False)
        tech_momentum = float(qqq["Close"].pct_change(periods=5).iloc[-1])
        
        return {
            'vix': vix_current,
            'treasury_yield': treasury_yield,
            'dollar_index': dollar_index,
            'spy_momentum': spy_momentum,
            'tech_momentum': tech_momentum
        }
    except:
        # Default values if data unavailable
        return {
            'vix': 20.0,
            'treasury_yield': 4.0,
            'dollar_index': 100.0,
            'spy_momentum': 0.0,
            'tech_momentum': 0.0
        }

def get_fundamental_data(ticker):
    """Get fundamental ratios for the stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'pe_ratio': info.get('trailingPE', 20.0) or 20.0,
            'pb_ratio': info.get('priceToBook', 3.0) or 3.0,
            'debt_to_equity': info.get('debtToEquity', 50.0) or 50.0,
            'roe': info.get('returnOnEquity', 0.15) or 0.15,
            'revenue_growth': info.get('revenueGrowth', 0.05) or 0.05,
            'profit_margin': info.get('profitMargins', 0.1) or 0.1,
            'beta': info.get('beta', 1.0) or 1.0,
        }
    except:
        # Default values if data unavailable
        return {
            'pe_ratio': 20.0,
            'pb_ratio': 3.0,
            'debt_to_equity': 50.0,
            'roe': 0.15,
            'revenue_growth': 0.05,
            'profit_margin': 0.1,
            'beta': 1.0,
        }

def get_sector_performance(ticker):
    """Get sector ETF performance"""
    sector_etfs = {
        'AAPL': 'XLK',   # Technology
        'NVDA': 'XLK',   # Technology  
        'AMZN': 'XLY',   # Consumer Discretionary
        'GOOGL': 'XLK',  # Technology
        'TSLA': 'XLY',   # Consumer Discretionary
    }
    
    try:
        etf_symbol = sector_etfs.get(ticker, 'SPY')
        etf_data = yf.download(etf_symbol, period="30d", progress=False)
        sector_momentum = float(etf_data["Close"].pct_change(periods=5).iloc[-1])
        return sector_momentum
    except:
        return 0.0

def fetch_stock_features(ticker):
    """ Fetch stock data and compute all technical indicators + enhanced features """
    # Get today's date and one year ago
    today = datetime.date.today()
    last_year = today - datetime.timedelta(days=730)  # 2 years of data
    df = yf.download(ticker, start=last_year.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))
    
    # Fix MultiIndex columns issue when downloading single ticker
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)
    
    # Get market context (same for all stocks on same date)
    print(f"   Getting market context...")
    market_data = get_market_context()
    
    # Get fundamental data
    print(f"   Getting fundamental data...")
    fundamental_data = get_fundamental_data(ticker)
    
    # Get sector performance
    print(f"   Getting sector data...")
    sector_momentum = get_sector_performance(ticker)
    
    # Basic price features
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    
    # Price momentum
    df["Return"] = df["Close"].pct_change()
    df["Return_5"] = df["Close"].pct_change(periods=5)
    df["Return_10"] = df["Close"].pct_change(periods=10)
    
    # Volatility measures
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["Volatility_20"] = df["Close"].rolling(window=20).std()
    
    # RSI
    df["RSI"] = compute_rsi(df["Close"])
    df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)
    df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
    
    # MACD
    macd_line, signal_line, histogram = compute_macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Histogram"] = histogram
    df["MACD_Bullish"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
    
    # Bollinger Bands
    upper_bb, lower_bb, bb_position = compute_bollinger_bands(df["Close"])
    df["BB_Upper"] = upper_bb
    df["BB_Lower"] = lower_bb
    df["BB_Position"] = bb_position
    # Fix for BB_Squeeze - ensure we're working with Series
    bb_width = (upper_bb - lower_bb) / df["SMA_20"]
    df["BB_Squeeze"] = (bb_width < 0.1).astype(int)

    
    # Stochastic Oscillator
    k_percent, d_percent = compute_stochastic(df["High"], df["Low"], df["Close"])
    df["Stoch_K"] = k_percent
    df["Stoch_D"] = d_percent
    df["Stoch_Oversold"] = (df["Stoch_K"] < 20).astype(int)
    df["Stoch_Overbought"] = (df["Stoch_K"] > 80).astype(int)
    
    # Average True Range (Volatility)
    df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"])
    
    # Volume indicators (if volume data available)
    if "Volume" in df.columns:
        df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
        df["Price_Volume"] = df["Close"] * df["Volume"]
        df["High_Volume"] = (df["Volume_Ratio"] > 1.5).astype(int)
    else:
        df["Volume_Ratio"] = 1
        df["Price_Volume"] = df["Close"]
        df["High_Volume"] = 0
    
    # Trend indicators
    df["Price_Above_SMA20"] = (df["Close"] > df["SMA_20"]).astype(int)
    df["Price_Above_SMA50"] = (df["Close"] > df["SMA_50"]).astype(int)
    df["SMA_Trend"] = (df["SMA_10"] > df["SMA_20"]).astype(int)
    
    # Support/Resistance levels
    df["High_20"] = df["High"].rolling(window=20).max()
    df["Low_20"] = df["Low"].rolling(window=20).min()
    df["Near_Resistance"] = (df["Close"] / df["High_20"] > 0.95).astype(int)
    df["Near_Support"] = (df["Close"] / df["Low_20"] < 1.05).astype(int)
    
    # Price gaps
    df["Gap_Up"] = (df["Open"] > df["Close"].shift(1) * 1.02).astype(int)
    df["Gap_Down"] = (df["Open"] < df["Close"].shift(1) * 0.98).astype(int)
    
    # 🚀 NEW ENHANCED FEATURES 🚀
    
    # Market Context Features (same for all rows since we get current market state)
    df["VIX"] = market_data['vix']
    df["Treasury_Yield"] = market_data['treasury_yield'] 
    df["Dollar_Index"] = market_data['dollar_index']
    df["SPY_Momentum"] = market_data['spy_momentum']
    df["Tech_Momentum"] = market_data['tech_momentum']
    
    # Market condition indicators
    df["Low_VIX"] = (df["VIX"] < 20).astype(int)  # Low fear = bullish
    df["High_VIX"] = (df["VIX"] > 30).astype(int)  # High fear = bearish
    df["Rising_Rates"] = (df["Treasury_Yield"] > 4.0).astype(int)  # High rates = pressure on growth
    df["Strong_Dollar"] = (df["Dollar_Index"] > 100).astype(int)  # Strong dollar = headwind
    df["Bull_Market"] = (df["SPY_Momentum"] > 0.02).astype(int)  # Strong S&P momentum
    
    # Fundamental Features (same for all rows since we get current fundamental state)
    df["PE_Ratio"] = fundamental_data['pe_ratio']
    df["PB_Ratio"] = fundamental_data['pb_ratio']
    df["Debt_to_Equity"] = fundamental_data['debt_to_equity']
    df["ROE"] = fundamental_data['roe']
    df["Revenue_Growth"] = fundamental_data['revenue_growth']
    df["Profit_Margin"] = fundamental_data['profit_margin']
    df["Beta"] = fundamental_data['beta']
    
    # Fundamental condition indicators
    df["Expensive_PE"] = (df["PE_Ratio"] > 25).astype(int)  # High valuation
    df["Cheap_PE"] = (df["PE_Ratio"] < 15).astype(int)  # Low valuation
    df["High_Growth"] = (df["Revenue_Growth"] > 0.1).astype(int)  # Fast growing
    df["High_ROE"] = (df["ROE"] > 0.2).astype(int)  # Very profitable
    df["Low_Debt"] = (df["Debt_to_Equity"] < 50).astype(int)  # Conservative debt
    df["High_Beta"] = (df["Beta"] > 1.2).astype(int)  # More volatile than market
    
    # Sector Performance
    df["Sector_Momentum"] = sector_momentum
    df["Strong_Sector"] = (df["Sector_Momentum"] > 0.02).astype(int)  # Sector outperforming
    
    # Combined signals
    df["Bullish_Market_Setup"] = (df["Low_VIX"] & df["Bull_Market"] & df["Strong_Sector"]).astype(int)
    df["Bearish_Market_Setup"] = (df["High_VIX"] & (df["SPY_Momentum"] < -0.02)).astype(int)
    df["Quality_Growth"] = (df["High_Growth"] & df["High_ROE"] & df["Low_Debt"]).astype(int)
    
    df = df.dropna()
    
    # Updated feature list with new enhanced features
    feature_columns = [
        # Original technical features
        "Close", "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "Return", "Return_5", "Return_10", "Volatility", "Volatility_20",
        "RSI", "RSI_Oversold", "RSI_Overbought",
        "MACD", "MACD_Signal", "MACD_Histogram", "MACD_Bullish",
        "BB_Position", "BB_Squeeze",
        "Stoch_K", "Stoch_D", "Stoch_Oversold", "Stoch_Overbought",
        "ATR", "Volume_Ratio", "High_Volume",
        "Price_Above_SMA20", "Price_Above_SMA50", "SMA_Trend",
        "Near_Resistance", "Near_Support", "Gap_Up", "Gap_Down",
        
        # NEW: Market context features
        "VIX", "Treasury_Yield", "Dollar_Index", "SPY_Momentum", "Tech_Momentum",
        "Low_VIX", "High_VIX", "Rising_Rates", "Strong_Dollar", "Bull_Market",
        
        # NEW: Fundamental features  
        "PE_Ratio", "PB_Ratio", "Debt_to_Equity", "ROE", "Revenue_Growth", 
        "Profit_Margin", "Beta", "Expensive_PE", "Cheap_PE", "High_Growth",
        "High_ROE", "Low_Debt", "High_Beta",
        
        # NEW: Sector and combined features
        "Sector_Momentum", "Strong_Sector", "Bullish_Market_Setup", 
        "Bearish_Market_Setup", "Quality_Growth"
    ]
    
    return df[feature_columns]

def label_data(df):
    future_returns = df["Close"].pct_change(periods=5).shift(-5)
    df["Label"] = future_returns.apply(lambda x: "buy" if x > 0.03 else ("sell" if x < -0.03 else "hold"))
    return df.dropna()

def train_model(ticker):
    """
    Train a Random Forest model for stock price prediction
    
    Args:
        ticker (str): Stock symbol (e.g., 'AAPL',...., 'TSLA')
    """
    # Print status message to track progress
    print(f"Fetching data for {ticker}...")
    
    # Download and compute all technical indicators for the stock
    df = fetch_stock_features(ticker)
    
    # Add buy/sell/hold labels based on future 5-day returns
    df = label_data(df)
    
    # Separate features from labels
    # Get all column names except 'Label' column
    feature_columns = [col for col in df.columns if col != 'Label']
    
    # Extract feature data (X) - all technical indicators
    features = df[feature_columns]
    
    # Extract target labels (y) - buy/sell/hold decisions
    labels = df["Label"]
    
    # Display dataset information for analysis
    print(f"Number of features: {len(feature_columns)}")  # Show how many indicators we're using
    print(f"Feature columns: {feature_columns}")         # List all technical indicators
    print(f"Data shape: {features.shape}")               # Show rows x columns (samples x features)
    print(f"Label distribution:")                        # Show balance of buy/sell/hold labels
    print(labels.value_counts())

    # Normalize features to have mean=0 and std=1
    # This prevents features with large values from dominating the model
    scaler = StandardScaler()
    X = scaler.fit_transform(features)  # Fit scaler and transform features
    y = labels                          # Keep labels as-is (categorical)

    # Split data into training and testing sets
    # test_size=0.2 means 80% training, 20% testing
    # stratify=y ensures balanced representation of buy/sell/hold in both sets
    # random_state=42 makes the split reproducible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create Random Forest classifier with optimized parameters
    # n_estimators=200: Use 200 decision trees for better accuracy
    # max_depth=10: Limit tree depth to prevent overfitting
    # min_samples_split=5: Require at least 5 samples to split a node
    # random_state=42: Make results reproducible
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    
    # Train the model on training data
    clf.fit(X_train, y_train)

    # Make predictions on test set (unseen data)
    preds = clf.predict(X_test)
    
    # Calculate overall accuracy (% of correct predictions)
    accuracy = accuracy_score(y_test, preds)
    
    # Display model performance metrics
    print(f"\nModel performance on test data:")
    print(f"Accuracy: {accuracy:.4f}")                    # Overall accuracy percentage
    print("\nClassification Report:")                     # Precision, recall, F1-score for each class
    print(classification_report(y_test, preds))
    
    print("\nConfusion Matrix:")                          # Matrix showing actual vs predicted labels
    print(confusion_matrix(y_test, preds))
    
    # Analyze which features are most important for predictions
    feature_importance = pd.DataFrame({
        'feature': feature_columns,                        # List of all feature names
        'importance': clf.feature_importances_             # Importance scores from Random Forest
    }).sort_values('importance', ascending=False)          # Sort by importance (highest first)
    
    # Display top 10 most influential technical indicators
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    # Save trained model components for later use in models folder
    joblib.dump(clf, f"models/{ticker}_model.pkl")               # Save trained Random Forest model
    joblib.dump(scaler, f"models/{ticker}_scaler.pkl")           # Save feature scaler (for consistent preprocessing)
    joblib.dump(feature_columns, f"models/{ticker}_features.pkl") # Save feature list (for consistent feature order)
    
    # Confirm successful save
    print(f"\nModel, scaler, and feature list saved for {ticker}")

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA"]  # Test all stocks with enhanced features
    for ticker in tickers:
        train_model(ticker)
