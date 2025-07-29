#!/usr/bin/env python3
"""
Test script for the Stock Advisor API
"""
import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Stock Advisor API")
    print("=" * 50)
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test market summary
    try:
        response = requests.get(f"{base_url}/api/market-summary")
        print(f"✅ Market summary: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Predictions: {len(data.get('predictions', []))}")
            print(f"   Market sentiment: {data.get('market_sentiment')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Market summary failed: {e}")
    
    # Test individual stock prediction
    test_ticker = "AAPL"
    try:
        response = requests.get(f"{base_url}/api/predict/{test_ticker}")
        print(f"✅ Stock prediction ({test_ticker}): {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Confidence: {data.get('confidence', 0):.2%}")
            print(f"   Current price: ${data.get('current_price', 0):.2f}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Stock prediction failed: {e}")
    
    print("\n🎉 API test completed!")

if __name__ == "__main__":
    test_api()
