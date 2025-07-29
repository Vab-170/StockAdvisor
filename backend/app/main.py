from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

# Import our ML prediction modules
from .prediction_service import PredictionService
from .models.schemas import (
    StockPrediction, 
    MarketSummary, 
    AnalysisRequest,
    HealthCheck
)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StockAdvisor API",
    description="AI-powered stock prediction and analysis API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()

# Supported stocks
SUPPORTED_STOCKS = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA"]

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )

@app.get("/api/stocks")
async def get_supported_stocks():
    """Get list of supported stock tickers"""
    return {
        "stocks": SUPPORTED_STOCKS,
        "count": len(SUPPORTED_STOCKS),
        "last_updated": datetime.now()
    }

@app.get("/api/predict/{ticker}", response_model=StockPrediction)
async def predict_stock(ticker: str):
    """Get prediction for a specific stock"""
    ticker = ticker.upper()
    
    if ticker not in SUPPORTED_STOCKS:
        raise HTTPException(
            status_code=404, 
            detail=f"Stock {ticker} not supported. Supported stocks: {SUPPORTED_STOCKS}"
        )
    
    try:
        prediction = await prediction_service.predict(ticker)
        return prediction
    except Exception as e:
        logger.error(f"Error predicting {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/market-summary", response_model=MarketSummary)
async def get_market_summary():
    """Get market summary for all supported stocks"""
    try:
        summary = await prediction_service.get_market_summary(SUPPORTED_STOCKS)
        return summary
    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market summary failed: {str(e)}")

@app.post("/api/analyze", response_model=List[StockPrediction])
async def analyze_stocks(request: AnalysisRequest):
    """Analyze custom list of stocks"""
    unsupported = [ticker for ticker in request.tickers if ticker.upper() not in SUPPORTED_STOCKS]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported stocks: {unsupported}. Supported: {SUPPORTED_STOCKS}"
        )
    
    try:
        predictions = []
        for ticker in request.tickers:
            prediction = await prediction_service.predict(ticker.upper())
            predictions.append(prediction)
        return predictions
    except Exception as e:
        logger.error(f"Error analyzing stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
