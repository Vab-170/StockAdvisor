import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

def get_vix_data():
    """Get VIX (fear index) data - market volatility indicator"""
    try:
        vix = yf.download("^VIX", period="1y")
        return vix["Close"].iloc[-1]  # Latest VIX value
    except:
        return None

def get_treasury_yield():
    """Get 10-year treasury yield - risk-free rate"""
    try:
        treasury = yf.download("^TNX", period="1y")
        return treasury["Close"].iloc[-1]  # Latest 10-year yield
    except:
        return None

def get_dollar_index():
    """Get US Dollar Index (DXY) - currency strength"""
    try:
        dxy = yf.download("DX-Y.NYB", period="1y")
        return dxy["Close"].iloc[-1]  # Latest DXY value
    except:
        return None

def get_sector_etf_performance(ticker):
    """Get sector performance relative to stock"""
    sector_etfs = {
        'AAPL': 'XLK',   # Technology
        'NVDA': 'XLK',   # Technology  
        'AMZN': 'XLY',   # Consumer Discretionary
        'GOOGL': 'XLK',  # Technology
        'TSLA': 'XLY',   # Consumer Discretionary
    }
    
    try:
        etf_symbol = sector_etfs.get(ticker)
        if etf_symbol:
            etf_data = yf.download(etf_symbol, period="1mo")
            etf_return = etf_data["Close"].pct_change(periods=5).iloc[-1]
            return float(etf_return)  # Convert to float
    except:
        return 0.0
    return 0.0

def get_fundamental_ratios(ticker):
    """Get basic fundamental data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'profit_margin': info.get('profitMargins', 0),
            'market_cap': info.get('marketCap', 0),
            'beta': info.get('beta', 1.0),
        }
        return fundamentals
    except:
        # Return default values if data unavailable
        return {
            'pe_ratio': 0, 'pb_ratio': 0, 'debt_to_equity': 0,
            'roe': 0, 'revenue_growth': 0, 'profit_margin': 0,
            'market_cap': 0, 'beta': 1.0
        }

def get_analyst_data(ticker):
    """Get analyst recommendations from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        if recommendations is not None and len(recommendations) > 0:
            latest = recommendations.iloc[-1]
            # Convert recommendations to numerical score
            score = (latest.get('strongBuy', 0) * 5 + 
                    latest.get('buy', 0) * 4 + 
                    latest.get('hold', 0) * 3 + 
                    latest.get('sell', 0) * 2 + 
                    latest.get('strongSell', 0) * 1)
            total = sum([latest.get(col, 0) for col in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
            return score / total if total > 0 else 3.0  # 3 = neutral
        return 3.0
    except:
        return 3.0  # Default to neutral

def get_earnings_surprise(ticker):
    """Get recent earnings surprise data"""
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings
        if earnings is not None and len(earnings) > 0:
            # Calculate earnings surprise (if available)
            latest_earnings = earnings.iloc[0]['Earnings']
            return latest_earnings if pd.notna(latest_earnings) else 0
        return 0
    except:
        return 0

def get_enhanced_market_data():
    """Get market-wide indicators"""
    market_data = {}
    
    # Market fear/greed indicators
    market_data['vix'] = get_vix_data()
    market_data['treasury_10y'] = get_treasury_yield()
    market_data['dollar_index'] = get_dollar_index()
    
    # Major indices performance
    try:
        spy = yf.download("SPY", period="1mo")
        market_data['spy_return_5d'] = spy["Close"].pct_change(periods=5).iloc[-1]
        
        qqq = yf.download("QQQ", period="1mo")
        market_data['qqq_return_5d'] = qqq["Close"].pct_change(periods=5).iloc[-1]
        
        # Small cap vs large cap performance
        iwm = yf.download("IWM", period="1mo")  # Small cap
        market_data['small_cap_performance'] = iwm["Close"].pct_change(periods=5).iloc[-1]
        
    except:
        market_data['spy_return_5d'] = 0
        market_data['qqq_return_5d'] = 0
        market_data['small_cap_performance'] = 0
    
    return market_data

def test_enhanced_data():
    """Test all enhanced data sources"""
    print("🔍 Testing Enhanced Data Sources")
    print("=" * 50)
    
    # Test market data
    print("\n📊 Market Indicators:")
    market = get_enhanced_market_data()
    for key, value in market.items():
        print(f"   {key}: {value}")
    
    # Test stock-specific data
    ticker = "AAPL"
    print(f"\n📈 {ticker} Fundamental Data:")
    fundamentals = get_fundamental_ratios(ticker)
    for key, value in fundamentals.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 {ticker} Additional Metrics:")
    print(f"   Analyst Score: {get_analyst_data(ticker)}")
    print(f"   Sector Performance: {get_sector_etf_performance(ticker):.3f}")
    print(f"   Earnings Data: {get_earnings_surprise(ticker)}")

if __name__ == "__main__":
    test_enhanced_data()
