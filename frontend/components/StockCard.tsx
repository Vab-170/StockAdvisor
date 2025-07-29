import { StockPrediction } from '../lib/api'
import { TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react'
import { formatCurrency, formatPercentage } from '../lib/utils'

interface StockCardProps {
  prediction: StockPrediction
}

export function StockCard({ prediction }: StockCardProps) {
  const getPredictionIcon = () => {
    switch (prediction.prediction) {
      case 'buy':
        return <TrendingUp className="h-5 w-5 text-success-600" />
      case 'sell':
        return <TrendingDown className="h-5 w-5 text-danger-600" />
      case 'hold':
        return <Minus className="h-5 w-5 text-warning-600" />
    }
  }

  const getPredictionColor = () => {
    switch (prediction.prediction) {
      case 'buy':
        return 'prediction-buy'
      case 'sell':
        return 'prediction-sell'
      case 'hold':
        return 'prediction-hold'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-success-600'
    if (confidence >= 0.6) return 'text-warning-600'
    return 'text-danger-600'
  }

  return (
    <div className="card hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            {prediction.ticker}
          </h3>
          <p className="text-sm text-gray-500">
            {formatCurrency(prediction.current_price)}
          </p>
        </div>
        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border ${getPredictionColor()}`}>
          {getPredictionIcon()}
          <span className="font-medium capitalize">
            {prediction.prediction}
          </span>
        </div>
      </div>

      <div className="space-y-3 mb-4">
        <div className="flex justify-between">
          <span className="text-sm text-gray-600">Confidence</span>
          <span className={`text-sm font-medium ${getConfidenceColor(prediction.confidence)}`}>
            {formatPercentage(prediction.confidence)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-sm text-gray-600">RSI</span>
          <span className="text-sm font-mono">
            {prediction.metrics.rsi.toFixed(1)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-sm text-gray-600">Daily Return</span>
          <span className={`text-sm font-mono ${
            prediction.metrics.daily_return >= 0 ? 'text-success-600' : 'text-danger-600'
          }`}>
            {formatPercentage(prediction.metrics.daily_return)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-sm text-gray-600">Volume</span>
          <span className="text-sm font-mono">
            {prediction.metrics.volume_ratio.toFixed(2)}x
          </span>
        </div>
      </div>

      <div className="border-t pt-3">
        <div className="flex items-start space-x-2">
          <Activity className="h-4 w-4 text-gray-400 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-gray-600 leading-relaxed">
            {prediction.explanation}
          </p>
        </div>
      </div>
    </div>
  )
}
