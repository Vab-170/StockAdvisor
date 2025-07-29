import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface StockMetrics {
  close: number
  sma_10: number
  sma_20: number
  sma_50: number
  rsi: number
  macd: number
  daily_return: number
  volume_ratio: number
  volatility_20: number
}

export interface MarketContext {
  vix: number
  treasury_yield: number
  dollar_index: number
  spy_momentum: number
  tech_momentum: number
}

export interface StockPrediction {
  ticker: string
  prediction: 'buy' | 'sell' | 'hold'
  confidence: number
  current_price: number
  explanation: string
  metrics: StockMetrics
  market_context: MarketContext
  timestamp: string
  model_version: string
}

export interface MarketSummary {
  predictions: StockPrediction[]
  market_sentiment: 'bullish' | 'bearish' | 'neutral'
  market_context: MarketContext
  timestamp: string
  summary_stats: {
    buy: number
    sell: number
    hold: number
  }
}

// API Functions
export const getMarketSummary = async (): Promise<MarketSummary> => {
  const response = await api.get('/api/market-summary')
  return response.data
}

export const getStockPrediction = async (ticker: string): Promise<StockPrediction> => {
  const response = await api.get(`/api/predict/${ticker}`)
  return response.data
}

export const getSupportedStocks = async (): Promise<{ stocks: string[]; count: number }> => {
  const response = await api.get('/api/stocks')
  return response.data
}

export const analyzeStocks = async (tickers: string[]): Promise<StockPrediction[]> => {
  const response = await api.post('/api/analyze', { 
    tickers,
    include_explanation: true 
  })
  return response.data
}

export const getHealthCheck = async () => {
  const response = await api.get('/')
  return response.data
}
