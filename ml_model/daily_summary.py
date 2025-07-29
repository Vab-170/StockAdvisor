import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# OPTIONAL: Add your OpenAI key for natural language explanations
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI library not installed. Install with: pip install openai")
    OPENAI_AVAILABLE = False
except Exception as e:
    print(f"OpenAI setup issue: {e}")
    OPENAI_AVAILABLE = False

# Import all the feature computation functions from train_model
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
    """Get current market context indicators"""
    print("   Getting market context...")
    # VIX (Market Fear Index)
    vix = yf.download("^VIX", period="30d", progress=False)
    vix_current = float(vix["Close"].iloc[-1])
    
    # 10-Year Treasury Yield
    treasury = yf.download("^TNX", period="30d", progress=False)
    treasury_yield = float(treasury["Close"].iloc[-1])
    
    # US Dollar Index
    dxy = yf.download("DX-Y.NYB", period="30d", progress=False)
    dollar_index = float(dxy["Close"].iloc[-1])
    
    # S&P 500 momentum
    spy = yf.download("SPY", period="30d", progress=False)
    spy_momentum = float(spy["Close"].pct_change(periods=5).iloc[-1])
    
    # NASDAQ momentum (tech focus)
    qqq = yf.download("QQQ", period="30d", progress=False)
    tech_momentum = float(qqq["Close"].pct_change(periods=5).iloc[-1])
    
    return {
        'VIX': vix_current,
        'Treasury_Yield': treasury_yield, 
        'Dollar_Index': dollar_index,
        'SPY_Momentum': spy_momentum,
        'Tech_Momentum': tech_momentum,
        'Low_VIX': int(vix_current < 20),
        'High_VIX': int(vix_current > 30),
        'Rising_Rates': int(treasury_yield > 4.0),
        'Strong_Dollar': int(dollar_index > 100),
        'Bull_Market': int(spy_momentum > 0.02)
    }

def get_fundamental_data(ticker):
    """Get fundamental analysis data for the stock"""
    print("   Getting fundamental data...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key fundamental metrics with safe fallbacks
        pe_ratio = info.get('trailingPE', info.get('forwardPE', 20))
        pb_ratio = info.get('priceToBook', 3)
        debt_to_equity = info.get('debtToEquity', 50) / 100 if info.get('debtToEquity') else 0.5
        roe = info.get('returnOnEquity', 0.15)
        revenue_growth = info.get('revenueGrowth', 0.05)
        profit_margin = info.get('profitMargins', 0.1)
        beta = info.get('beta', 1.0)
        
        return {
            'PE_Ratio': pe_ratio,
            'PB_Ratio': pb_ratio,
            'Debt_to_Equity': debt_to_equity,
            'ROE': roe,
            'Revenue_Growth': revenue_growth,
            'Profit_Margin': profit_margin,
            'Beta': beta,
            'Expensive_PE': int(pe_ratio > 25),
            'Cheap_PE': int(pe_ratio < 15),
            'High_Growth': int(revenue_growth > 0.15),
            'High_ROE': int(roe > 0.15),
            'Low_Debt': int(debt_to_equity < 0.3),
            'High_Beta': int(beta > 1.3)
        }
    except Exception as e:
        print(f"   Warning: Could not get fundamental data: {e}")
        # Return default values if API fails
        return {
            'PE_Ratio': 20, 'PB_Ratio': 3, 'Debt_to_Equity': 0.5, 'ROE': 0.15,
            'Revenue_Growth': 0.05, 'Profit_Margin': 0.1, 'Beta': 1.0,
            'Expensive_PE': 0, 'Cheap_PE': 0, 'High_Growth': 0,
            'High_ROE': 0, 'Low_Debt': 1, 'High_Beta': 0
        }

def get_sector_performance(ticker):
    """Get sector performance relative to market"""
    print("   Getting sector data...")
    # Map tickers to sector ETFs for relative performance
    sector_map = {
        'AAPL': 'XLK',   # Technology
        'NVDA': 'XLK',   # Technology  
        'AMZN': 'XLY',   # Consumer Discretionary
        'GOOGL': 'XLK',  # Technology
        'TSLA': 'XLY'    # Consumer Discretionary
    }
    
    try:
        etf_symbol = sector_map.get(ticker, 'SPY')  # Default to SPY if not found
        etf_data = yf.download(etf_symbol, period="30d", progress=False)
        sector_momentum = float(etf_data["Close"].pct_change(periods=5).iloc[-1])
        
        return {
            'Sector_Momentum': sector_momentum,
            'Strong_Sector': int(sector_momentum > 0.01)
        }
    except Exception as e:
        print(f"   Warning: Could not get sector data: {e}")
        return {'Sector_Momentum': 0, 'Strong_Sector': 0}

def get_latest_features(ticker):
    """Get the latest feature data for prediction (matches enhanced training model exactly)"""
    end = datetime.today()
    start = end - timedelta(days=90)  # Need more data for 50-day SMA and other indicators
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    
    # Fix MultiIndex columns issue when downloading single ticker
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)
    
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
    
    # Get enhanced features (market context, fundamentals, sector data)
    market_data = get_market_context()
    fundamental_data = get_fundamental_data(ticker)
    sector_data = get_sector_performance(ticker)
    
    # Add market context features (constant for all rows since they're current market conditions)
    for key, value in market_data.items():
        df[key] = value
    
    # Add fundamental features (constant for all rows since they're company-specific)
    for key, value in fundamental_data.items():
        df[key] = value
        
    # Add sector features
    for key, value in sector_data.items():
        df[key] = value
    
    # Add combined signal features
    df['Bullish_Market_Setup'] = (
        (df['Low_VIX'] == 1) & 
        (df['Bull_Market'] == 1) & 
        (df['Strong_Sector'] == 1)
    ).astype(int)
    
    df['Bearish_Market_Setup'] = (
        (df['High_VIX'] == 1) & 
        (df['Rising_Rates'] == 1) & 
        (df['Strong_Dollar'] == 1)
    ).astype(int)
    
    df['Quality_Growth'] = (
        (df['High_ROE'] == 1) & 
        (df['High_Growth'] == 1) & 
        (df['Low_Debt'] == 1)
    ).astype(int)
    
    df = df.dropna()
    
    # Define the same feature columns as in enhanced training model (62 features)
    feature_columns = [
        # Original technical features (34)
        "Close", "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "Return", "Return_5", "Return_10", "Volatility", "Volatility_20",
        "RSI", "RSI_Oversold", "RSI_Overbought",
        "MACD", "MACD_Signal", "MACD_Histogram", "MACD_Bullish",
        "BB_Position", "BB_Squeeze",
        "Stoch_K", "Stoch_D", "Stoch_Oversold", "Stoch_Overbought",
        "ATR", "Volume_Ratio", "High_Volume",
        "Price_Above_SMA20", "Price_Above_SMA50", "SMA_Trend",
        "Near_Resistance", "Near_Support", "Gap_Up", "Gap_Down",
        # Market context features (10)
        "VIX", "Treasury_Yield", "Dollar_Index", "SPY_Momentum", "Tech_Momentum",
        "Low_VIX", "High_VIX", "Rising_Rates", "Strong_Dollar", "Bull_Market",
        # Fundamental features (13)
        "PE_Ratio", "PB_Ratio", "Debt_to_Equity", "ROE", "Revenue_Growth", "Profit_Margin", "Beta",
        "Expensive_PE", "Cheap_PE", "High_Growth", "High_ROE", "Low_Debt", "High_Beta",
        # Sector features (2)
        "Sector_Momentum", "Strong_Sector",
        # Combined signals (3)
        "Bullish_Market_Setup", "Bearish_Market_Setup", "Quality_Growth"
    ]
    
    return df[feature_columns].iloc[-1]  # return most recent row

def predict_today(ticker):
    """Make prediction using the trained model with all 62 features"""
    try:
        # Load the trained model components from models folder
        model = joblib.load(f"models/{ticker}_model.pkl")
        scaler = joblib.load(f"models/{ticker}_scaler.pkl")
        feature_columns = joblib.load(f"models/{ticker}_features.pkl")
        
        # Get latest feature data
        features = get_latest_features(ticker)
        
        # Ensure features are in the correct order
        X = features[feature_columns].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0].max()
        
        return prediction, features, confidence
    except Exception as e:
        return "error", str(e), 0.0

def explain_with_openai(ticker, features, prediction, confidence):
    """Generate human-readable explanation using OpenAI"""
    # Check if OpenAI is available and configured
    if not OPENAI_AVAILABLE:
        simple_explanation = generate_simple_explanation(ticker, features, prediction, confidence)
        return f"AI explanation unavailable (OpenAI library not installed). {simple_explanation}"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        simple_explanation = generate_simple_explanation(ticker, features, prediction, confidence)
        return f"AI explanation unavailable (API key not configured). {simple_explanation}"
    
    prompt = f"""
    A machine learning model predicted {prediction.upper()} for stock {ticker} today with {confidence*100:.1f}% confidence.
    
    Key technical indicators:
    - Close price: ${features['Close']:.2f}
    - 10-day SMA: ${features['SMA_10']:.2f}
    - 20-day SMA: ${features['SMA_20']:.2f}
    - 50-day SMA: ${features['SMA_50']:.2f}
    - RSI: {features['RSI']:.1f}
    - MACD: {features['MACD']:.3f}
    - Daily Return: {features['Return']*100:.2f}%
    - Volatility (20-day): {features['Volatility_20']:.2f}
    - Volume Ratio: {features['Volume_Ratio']:.2f}
    - Price above 20-day SMA: {'Yes' if features['Price_Above_SMA20'] else 'No'}
    - Price above 50-day SMA: {'Yes' if features['Price_Above_SMA50'] else 'No'}

    Write a 2-3 sentence explanation in plain English why the model might have made this {prediction} recommendation.
    Focus on the most relevant indicators for this decision.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the more cost-effective model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback explanation based on simple rules
        simple_explanation = generate_simple_explanation(ticker, features, prediction, confidence)
        return f"AI explanation unavailable ({str(e)[:50]}...). {simple_explanation}"

def generate_simple_explanation(ticker, features, prediction, confidence):
    """Generate a simple rule-based explanation as fallback"""
    explanations = []
    
    if features['Price_Above_SMA20'] and features['Price_Above_SMA50']:
        explanations.append("price is above both 20-day and 50-day moving averages (bullish trend)")
    elif not features['Price_Above_SMA20'] and not features['Price_Above_SMA50']:
        explanations.append("price is below both 20-day and 50-day moving averages (bearish trend)")
    
    if features['RSI'] > 70:
        explanations.append("RSI indicates overbought conditions")
    elif features['RSI'] < 30:
        explanations.append("RSI indicates oversold conditions")
    
    if features['Volume_Ratio'] > 1.5:
        explanations.append("high trading volume suggests strong interest")
    
    if features['MACD'] > features['MACD_Signal']:
        explanations.append("MACD is bullish")
    else:
        explanations.append("MACD is bearish")
    
    if explanations:
        return f"Model suggests {prediction.upper()} because " + " and ".join(explanations[:2]) + "."
    else:
        return f"Model suggests {prediction.upper()} with {confidence*100:.1f}% confidence."

def run_summary(tickers):
    """Generate daily stock predictions summary"""
    print(f"\n📈 Stock Advisory Summary - {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    for ticker in tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        prediction, features, confidence = predict_today(ticker)
        
        if prediction == "error":
            print(f"{ticker}: ❌ Error - {features}")
            continue

        # Determine emoji for prediction
        emoji = {"buy": "🚀", "sell": "📉", "hold": "⏸️"}.get(prediction, "❓")
        
        print(f"\n{ticker}: {emoji} {prediction.upper()} (Confidence: {confidence*100:.1f}%)")
        print(f"💰 Current Price: ${features['Close']:.2f}")
        print(f"📊 Key Metrics:")
        print(f"   • 10-day SMA: ${features['SMA_10']:.2f}")
        print(f"   • 20-day SMA: ${features['SMA_20']:.2f}")
        print(f"   • RSI: {features['RSI']:.1f}")
        print(f"   • Daily Return: {features['Return']*100:.2f}%")
        print(f"   • Volume Ratio: {features['Volume_Ratio']:.2f}x")
        
        # Generate explanation
        explanation = explain_with_openai(ticker, features, prediction, confidence)
        print(f"🧠 Analysis: {explanation}")
        print("-" * 70)

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA"]  # Updated to match training model
    run_summary(tickers)
