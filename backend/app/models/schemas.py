from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any

class StockMetrics(BaseModel):
    close: float
    sma_10: float
    sma_20: float
    sma_50: float
    rsi: float
    macd: float
    daily_return: float
    volume_ratio: float
    volatility_20: float

class MarketContext(BaseModel):
    vix: float
    treasury_yield: float
    dollar_index: float
    spy_momentum: float
    tech_momentum: float

class StockPrediction(BaseModel):
    ticker: str
    prediction: str  # buy, sell, hold
    confidence: float
    current_price: float
    explanation: str
    metrics: StockMetrics
    market_context: MarketContext
    timestamp: datetime
    model_version: str = "v1.0"

class MarketSummary(BaseModel):
    predictions: List[StockPrediction]
    market_sentiment: str  # bullish, bearish, neutral
    market_context: MarketContext
    timestamp: datetime
    summary_stats: Dict[str, int]  # count of buy/sell/hold

class AnalysisRequest(BaseModel):
    tickers: List[str]
    include_explanation: bool = True

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str
