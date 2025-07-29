import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional
from .models.schemas import StockPrediction, MarketSummary, StockMetrics, MarketContext

# Import OpenAI safely
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for stock predictions using trained ML models"""
    
    def __init__(self):
        # Get absolute path to models directory
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.supported_tickers = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA"]
        
        # Initialize OpenAI if available
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.openai_client = None
            logger.warning("OpenAI not available for explanations")
    
    async def predict(self, ticker: str) -> StockPrediction:
        """Get prediction for a single stock"""
        try:
            # Get latest features
            features = await self._get_latest_features(ticker)
            
            # Load model and make prediction
            prediction, confidence = await self._make_prediction(ticker, features)
            
            # Get market context
            market_context = await self._get_market_context()
            
            # Generate explanation
            explanation = await self._generate_explanation(
                ticker, features, prediction, confidence
            )
            
            # Extract metrics
            metrics = self._extract_metrics(features)
            
            return StockPrediction(
                ticker=ticker,
                prediction=prediction,
                confidence=confidence,
                current_price=float(features['Close']),
                explanation=explanation,
                metrics=metrics,
                market_context=market_context,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting {ticker}: {str(e)}")
            raise
    
    async def get_market_summary(self, tickers: List[str]) -> MarketSummary:
        """Get market summary for multiple stocks"""
        predictions = []
        
        for ticker in tickers:
            try:
                prediction = await self.predict(ticker)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict {ticker}: {str(e)}")
                continue
        
        # Calculate summary stats
        summary_stats = {
            "buy": sum(1 for p in predictions if p.prediction == "buy"),
            "sell": sum(1 for p in predictions if p.prediction == "sell"),
            "hold": sum(1 for p in predictions if p.prediction == "hold")
        }
        
        # Determine market sentiment
        market_sentiment = self._determine_market_sentiment(summary_stats)
        
        # Get market context (use first prediction's context)
        market_context = predictions[0].market_context if predictions else await self._get_market_context()
        
        return MarketSummary(
            predictions=predictions,
            market_sentiment=market_sentiment,
            market_context=market_context,
            timestamp=datetime.now(),
            summary_stats=summary_stats
        )
    
    async def _get_latest_features(self, ticker: str) -> pd.Series:
        """Get latest features for prediction"""
        end = datetime.today()
        start = end - timedelta(days=90)
        
        # Download stock data
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # Handle MultiIndex columns
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Add market context
        market_data = await self._get_market_context_data()
        for key, value in market_data.items():
            df[key] = value
        
        # Add fundamental data
        fundamental_data = await self._get_fundamental_data(ticker)
        for key, value in fundamental_data.items():
            df[key] = value
        
        # Add sector data
        sector_data = await self._get_sector_data(ticker)
        for key, value in sector_data.items():
            df[key] = value
        
        # Add combined signals
        df = self._add_combined_signals(df)
        
        return df.dropna().iloc[-1]
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Moving averages
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()
        
        # Returns and volatility
        df["Return"] = df["Close"].pct_change()
        df["Return_5"] = df["Close"].pct_change(periods=5)
        df["Return_10"] = df["Close"].pct_change(periods=10)
        df["Volatility"] = df["Close"].rolling(window=10).std()
        df["Volatility_20"] = df["Close"].rolling(window=20).std()
        
        # RSI
        df["RSI"] = self._compute_rsi(df["Close"])
        df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)
        df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
        
        # MACD
        macd_line, signal_line, histogram = self._compute_macd(df["Close"])
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
        df["MACD_Histogram"] = histogram
        df["MACD_Bullish"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
        
        # Bollinger Bands
        upper_bb, lower_bb, bb_position = self._compute_bollinger_bands(df["Close"])
        df["BB_Position"] = bb_position
        bb_width = (upper_bb - lower_bb) / df["SMA_20"]
        df["BB_Squeeze"] = (bb_width < 0.1).astype(int)
        
        # Stochastic
        k_percent, d_percent = self._compute_stochastic(df["High"], df["Low"], df["Close"])
        df["Stoch_K"] = k_percent
        df["Stoch_D"] = d_percent
        df["Stoch_Oversold"] = (df["Stoch_K"] < 20).astype(int)
        df["Stoch_Overbought"] = (df["Stoch_K"] > 80).astype(int)
        
        # ATR
        df["ATR"] = self._compute_atr(df["High"], df["Low"], df["Close"])
        
        # Volume indicators
        if "Volume" in df.columns:
            df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
            df["High_Volume"] = (df["Volume_Ratio"] > 1.5).astype(int)
        else:
            df["Volume_Ratio"] = 1
            df["High_Volume"] = 0
        
        # Trend indicators
        df["Price_Above_SMA20"] = (df["Close"] > df["SMA_20"]).astype(int)
        df["Price_Above_SMA50"] = (df["Close"] > df["SMA_50"]).astype(int)
        df["SMA_Trend"] = (df["SMA_10"] > df["SMA_20"]).astype(int)
        
        # Support/Resistance
        df["High_20"] = df["High"].rolling(window=20).max()
        df["Low_20"] = df["Low"].rolling(window=20).min()
        df["Near_Resistance"] = (df["Close"] / df["High_20"] > 0.95).astype(int)
        df["Near_Support"] = (df["Close"] / df["Low_20"] < 1.05).astype(int)
        
        # Price gaps
        df["Gap_Up"] = (df["Open"] > df["Close"].shift(1) * 1.02).astype(int)
        df["Gap_Down"] = (df["Open"] < df["Close"].shift(1) * 0.98).astype(int)
        
        return df
    
    def _compute_rsi(self, series, period=14):
        """Compute RSI"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, series, fast=12, slow=26, signal=9):
        """Compute MACD"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _compute_bollinger_bands(self, series, period=20, std_dev=2):
        """Compute Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (series - lower_band) / (upper_band - lower_band)
        return upper_band, lower_band, bb_position
    
    def _compute_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Compute Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _compute_atr(self, high, low, close, period=14):
        """Compute Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    async def _get_market_context_data(self) -> Dict:
        """Get market context data"""
        try:
            # VIX
            vix = yf.download("^VIX", period="30d", progress=False)
            vix_current = float(vix["Close"].iloc[-1])
            
            # Treasury
            treasury = yf.download("^TNX", period="30d", progress=False)
            treasury_yield = float(treasury["Close"].iloc[-1])
            
            # Dollar Index
            dxy = yf.download("DX-Y.NYB", period="30d", progress=False)
            dollar_index = float(dxy["Close"].iloc[-1])
            
            # Market momentum
            spy = yf.download("SPY", period="30d", progress=False)
            spy_momentum = float(spy["Close"].pct_change(periods=5).iloc[-1])
            
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
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {
                'VIX': 20, 'Treasury_Yield': 4.0, 'Dollar_Index': 100,
                'SPY_Momentum': 0, 'Tech_Momentum': 0,
                'Low_VIX': 0, 'High_VIX': 0, 'Rising_Rates': 0,
                'Strong_Dollar': 0, 'Bull_Market': 0
            }
    
    async def _get_fundamental_data(self, ticker: str) -> Dict:
        """Get fundamental data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            pe_ratio = info.get('trailingPE', info.get('forwardPE', 20))
            pb_ratio = info.get('priceToBook', 3)
            debt_to_equity = info.get('debtToEquity', 50) / 100 if info.get('debtToEquity') else 0.5
            roe = info.get('returnOnEquity', 0.15)
            revenue_growth = info.get('revenueGrowth', 0.05)
            profit_margin = info.get('profitMargins', 0.1)
            beta = info.get('beta', 1.0)
            
            return {
                'PE_Ratio': pe_ratio, 'PB_Ratio': pb_ratio,
                'Debt_to_Equity': debt_to_equity, 'ROE': roe,
                'Revenue_Growth': revenue_growth, 'Profit_Margin': profit_margin,
                'Beta': beta, 'Expensive_PE': int(pe_ratio > 25),
                'Cheap_PE': int(pe_ratio < 15), 'High_Growth': int(revenue_growth > 0.15),
                'High_ROE': int(roe > 0.15), 'Low_Debt': int(debt_to_equity < 0.3),
                'High_Beta': int(beta > 1.3)
            }
        except Exception as e:
            logger.error(f"Error getting fundamental data for {ticker}: {e}")
            return {
                'PE_Ratio': 20, 'PB_Ratio': 3, 'Debt_to_Equity': 0.5, 'ROE': 0.15,
                'Revenue_Growth': 0.05, 'Profit_Margin': 0.1, 'Beta': 1.0,
                'Expensive_PE': 0, 'Cheap_PE': 0, 'High_Growth': 0,
                'High_ROE': 0, 'Low_Debt': 1, 'High_Beta': 0
            }
    
    async def _get_sector_data(self, ticker: str) -> Dict:
        """Get sector performance data"""
        sector_map = {
            'AAPL': 'XLK', 'NVDA': 'XLK', 'AMZN': 'XLY',
            'GOOGL': 'XLK', 'TSLA': 'XLY'
        }
        
        try:
            etf_symbol = sector_map.get(ticker, 'SPY')
            etf_data = yf.download(etf_symbol, period="30d", progress=False)
            sector_momentum = float(etf_data["Close"].pct_change(periods=5).iloc[-1])
            
            return {
                'Sector_Momentum': sector_momentum,
                'Strong_Sector': int(sector_momentum > 0.01)
            }
        except Exception as e:
            logger.error(f"Error getting sector data for {ticker}: {e}")
            return {'Sector_Momentum': 0, 'Strong_Sector': 0}
    
    def _add_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combined signal features"""
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
        
        return df
    
    async def _make_prediction(self, ticker: str, features: pd.Series) -> tuple:
        """Make prediction using trained model"""
        try:
            # Load model components
            model = joblib.load(f"{self.model_path}/{ticker}_model.pkl")
            scaler = joblib.load(f"{self.model_path}/{ticker}_scaler.pkl")
            feature_columns = joblib.load(f"{self.model_path}/{ticker}_features.pkl")
            
            # Prepare features
            X = features[feature_columns].values.reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            confidence = model.predict_proba(X_scaled)[0].max()
            
            return prediction, float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {e}")
            raise
    
    async def _get_market_context(self) -> MarketContext:
        """Get market context for response"""
        market_data = await self._get_market_context_data()
        return MarketContext(
            vix=market_data['VIX'],
            treasury_yield=market_data['Treasury_Yield'],
            dollar_index=market_data['Dollar_Index'],
            spy_momentum=market_data['SPY_Momentum'],
            tech_momentum=market_data['Tech_Momentum']
        )
    
    async def _generate_explanation(self, ticker: str, features: pd.Series, 
                                  prediction: str, confidence: float) -> str:
        """Generate explanation for prediction"""
        if self.openai_client:
            try:
                prompt = f"""
                A machine learning model predicted {prediction.upper()} for stock {ticker} 
                with {confidence*100:.1f}% confidence.
                
                Key indicators:
                - Price: ${features['Close']:.2f}
                - RSI: {features['RSI']:.1f}
                - MACD: {features['MACD']:.3f}
                - 20-day SMA: ${features['SMA_20']:.2f}
                
                Explain this {prediction} recommendation in 2-3 sentences.
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI explanation failed: {e}")
        
        # Fallback explanation
        return self._generate_simple_explanation(features, prediction, confidence)
    
    def _generate_simple_explanation(self, features: pd.Series, 
                                   prediction: str, confidence: float) -> str:
        """Generate simple rule-based explanation"""
        explanations = []
        
        if features['Price_Above_SMA20'] and features['Price_Above_SMA50']:
            explanations.append("price is above key moving averages")
        elif not features['Price_Above_SMA20']:
            explanations.append("price is below short-term average")
        
        if features['RSI'] > 70:
            explanations.append("RSI indicates overbought conditions")
        elif features['RSI'] < 30:
            explanations.append("RSI indicates oversold conditions")
        
        if explanations:
            return f"Model suggests {prediction.upper()} because " + " and ".join(explanations[:2]) + "."
        else:
            return f"Model suggests {prediction.upper()} with {confidence*100:.1f}% confidence."
    
    def _extract_metrics(self, features: pd.Series) -> StockMetrics:
        """Extract key metrics for response"""
        return StockMetrics(
            close=float(features['Close']),
            sma_10=float(features['SMA_10']),
            sma_20=float(features['SMA_20']),
            sma_50=float(features['SMA_50']),
            rsi=float(features['RSI']),
            macd=float(features['MACD']),
            daily_return=float(features['Return']),
            volume_ratio=float(features['Volume_Ratio']),
            volatility_20=float(features['Volatility_20'])
        )
    
    def _determine_market_sentiment(self, summary_stats: Dict[str, int]) -> str:
        """Determine overall market sentiment"""
        total = sum(summary_stats.values())
        if total == 0:
            return "neutral"
        
        buy_pct = summary_stats["buy"] / total
        sell_pct = summary_stats["sell"] / total
        
        if buy_pct > 0.6:
            return "bullish"
        elif sell_pct > 0.6:
            return "bearish"
        else:
            return "neutral"
