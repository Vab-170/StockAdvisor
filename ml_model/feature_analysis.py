import pandas as pd
from train_model import fetch_stock_features, get_market_context, get_fundamental_data

def analyze_new_features():
    """Analyze what the new enhanced features look like"""
    print("🔍 ENHANCED FEATURES ANALYSIS")
    print("=" * 60)
    
    # Get enhanced features for AAPL
    df = fetch_stock_features("AAPL")
    
    print(f"📊 DATASET INFO:")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   Data Points: {len(df)}")
    print(f"   Date Range: 2023-2025 (2 years)")
    
    # Show market context features
    print(f"\n🌍 CURRENT MARKET CONTEXT:")
    market_cols = ['VIX', 'Treasury_Yield', 'Dollar_Index', 'SPY_Momentum', 'Tech_Momentum']
    for col in market_cols:
        if col in df.columns:
            value = df[col].iloc[-1]
            print(f"   {col}: {value:.3f}")
    
    # Show market condition flags
    print(f"\n🚦 MARKET CONDITIONS:")
    condition_cols = ['Low_VIX', 'High_VIX', 'Rising_Rates', 'Strong_Dollar', 'Bull_Market']
    for col in condition_cols:
        if col in df.columns:
            value = df[col].iloc[-1]
            status = "✅ YES" if value == 1 else "❌ NO"
            print(f"   {col}: {status}")
    
    # Show fundamental features
    print(f"\n💰 FUNDAMENTAL METRICS:")
    fundamental_cols = ['PE_Ratio', 'PB_Ratio', 'ROE', 'Revenue_Growth', 'Profit_Margin', 'Beta']
    for col in fundamental_cols:
        if col in df.columns:
            value = df[col].iloc[-1]
            print(f"   {col}: {value:.3f}")
    
    # Show fundamental conditions
    print(f"\n📈 FUNDAMENTAL CONDITIONS:")
    fund_condition_cols = ['Expensive_PE', 'Cheap_PE', 'High_Growth', 'High_ROE', 'Low_Debt']
    for col in fund_condition_cols:
        if col in df.columns:
            value = df[col].iloc[-1]
            status = "✅ YES" if value == 1 else "❌ NO"
            print(f"   {col}: {status}")
    
    # Show combined signals
    print(f"\n🎯 COMBINED SIGNALS:")
    signal_cols = ['Bullish_Market_Setup', 'Bearish_Market_Setup', 'Quality_Growth', 'Strong_Sector']
    for col in signal_cols:
        if col in df.columns:
            value = df[col].iloc[-1]
            status = "🚀 ACTIVE" if value == 1 else "⏸️ INACTIVE"
            print(f"   {col}: {status}")
    
    # Feature importance comparison
    print(f"\n📊 FEATURE CATEGORIES:")
    technical_features = 34  # Original features
    market_features = 10    # Market context features  
    fundamental_features = 13  # Fundamental features
    combined_features = 5   # Combined signal features
    
    print(f"   Technical Indicators: {technical_features} features")
    print(f"   Market Context: {market_features} features")  
    print(f"   Fundamental Data: {fundamental_features} features")
    print(f"   Combined Signals: {combined_features} features")
    print(f"   TOTAL: {technical_features + market_features + fundamental_features + combined_features} features")

if __name__ == "__main__":
    analyze_new_features()
