"""
Simplified StockAdvisor API for Vercel deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from datetime import datetime
import json

# Simple FastAPI app for Vercel
app = FastAPI(
    title="StockAdvisor API",
    description="AI-powered stock prediction API (Vercel Edition)",
    version="1.0.0"
)

# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for demonstration (in production, replace with ML models)
SUPPORTED_STOCKS = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA"]

class StockPrediction(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    current_price: float
    explanation: str
    metrics: Dict[str, float]
    market_context: Dict[str, float]
    timestamp: datetime

class MarketSummary(BaseModel):
    predictions: List[StockPrediction]
    market_sentiment: str
    market_context: Dict[str, float]
    summary_stats: Dict[str, int]
    timestamp: datetime

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "StockAdvisor API is running",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/api/stocks")
async def get_supported_stocks():
    """Get list of supported stock tickers"""
    return {
        "stocks": SUPPORTED_STOCKS,
        "count": len(SUPPORTED_STOCKS),
        "last_updated": datetime.now()
    }

@app.get("/api/predict/{ticker}")
async def predict_stock(ticker: str):
    """Get prediction for a specific stock"""
    ticker = ticker.upper()
    
    if ticker not in SUPPORTED_STOCKS:
        raise HTTPException(
            status_code=404, 
            detail=f"Stock {ticker} not supported. Supported stocks: {SUPPORTED_STOCKS}"
        )
    
    # Mock prediction data (replace with actual ML model in production)
    mock_predictions = {
        "AAPL": {"prediction": "hold", "confidence": 0.76, "price": 214.05},
        "NVDA": {"prediction": "buy", "confidence": 0.82, "price": 117.45},
        "AMZN": {"prediction": "buy", "confidence": 0.74, "price": 178.32},
        "GOOGL": {"prediction": "hold", "confidence": 0.71, "price": 168.24},
        "TSLA": {"prediction": "sell", "confidence": 0.68, "price": 218.87}
    }
    
    mock_data = mock_predictions.get(ticker, mock_predictions["AAPL"])
    
    return StockPrediction(
        ticker=ticker,
        prediction=mock_data["prediction"],
        confidence=mock_data["confidence"],
        current_price=mock_data["price"],
        explanation=f"Based on technical analysis, the model suggests {mock_data['prediction'].upper()} for {ticker} with {mock_data['confidence']*100:.1f}% confidence.",
        metrics={
            "close": mock_data["price"],
            "sma_10": mock_data["price"] * 0.98,
            "sma_20": mock_data["price"] * 0.96,
            "sma_50": mock_data["price"] * 0.94,
            "rsi": 65.3,
            "macd": 2.56,
            "daily_return": 0.008,
            "volume_ratio": 1.2,
            "volatility_20": 2.46
        },
        market_context={
            "vix": 16.09,
            "treasury_yield": 4.33,
            "dollar_index": 98.87,
            "spy_momentum": 0.0099,
            "tech_momentum": 0.0102
        },
        timestamp=datetime.now()
    )

@app.get("/api/market-summary")
async def get_market_summary():
    """Get market summary for all supported stocks"""
    try:
        predictions = []
        for ticker in SUPPORTED_STOCKS:
            prediction = await predict_stock(ticker)
            predictions.append(prediction)
        
        # Calculate summary stats
        summary_stats = {
            "buy": sum(1 for p in predictions if p.prediction == "buy"),
            "sell": sum(1 for p in predictions if p.prediction == "sell"),
            "hold": sum(1 for p in predictions if p.prediction == "hold")
        }
        
        # Determine market sentiment
        if summary_stats["buy"] > summary_stats["sell"]:
            market_sentiment = "bullish"
        elif summary_stats["sell"] > summary_stats["buy"]:
            market_sentiment = "bearish"
        else:
            market_sentiment = "neutral"
        
        return MarketSummary(
            predictions=predictions,
            market_sentiment=market_sentiment,
            market_context={
                "vix": 16.09,
                "treasury_yield": 4.33,
                "dollar_index": 98.87,
                "spy_momentum": 0.0099,
                "tech_momentum": 0.0102
            },
            summary_stats=summary_stats,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market summary failed: {str(e)}")

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# For Vercel compatibility
handler = app
