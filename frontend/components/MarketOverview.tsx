import { MarketSummary } from '../lib/api'
import { cn, formatPercentage } from '../lib/utils'

interface MarketOverviewProps {
  data: MarketSummary
}

export function MarketOverview({ data }: MarketOverviewProps) {
  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-success-600'
    if (change < 0) return 'text-danger-600'
    return 'text-gray-500'
  }

  const getChangeIcon = (change: number) => {
    if (change > 0) return '↗'
    if (change < 0) return '↘'
    return '→'
  }

  const markets = [
    { name: 'S&P 500', value: data.market_context.spy_momentum, symbol: 'SPY' },
    { name: 'VIX', value: data.market_context.vix, symbol: 'VIX', isVix: true },
    { name: '10Y Treasury', value: data.market_context.treasury_yield, symbol: '10Y', isTreasury: true },
    { name: 'DXY', value: data.market_context.dollar_index, symbol: 'DXY' },
  ]

  return (
    <div className="card">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Market Overview
      </h2>
      
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {markets.map((market) => (
          <div key={market.symbol} className="text-center">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
              {market.name}
            </div>
            <div className={cn(
              "text-lg font-semibold",
              market.isVix 
                ? (market.value > 20 ? 'text-danger-600' : market.value > 15 ? 'text-warning-600' : 'text-success-600')
                : market.isTreasury
                ? 'text-gray-900'
                : getChangeColor(market.value)
            )}>
              {market.isTreasury || market.isVix 
                ? market.value.toFixed(2) + (market.isTreasury ? '%' : '')
                : (
                  <span className="flex items-center justify-center space-x-1">
                    <span>{getChangeIcon(market.value)}</span>
                    <span>{formatPercentage(Math.abs(market.value))}</span>
                  </span>
                )
              }
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Market Sentiment</span>
          <span className={cn(
            "font-medium",
            data.market_sentiment === 'bullish' ? 'text-success-600' :
            data.market_sentiment === 'bearish' ? 'text-danger-600' :
            'text-warning-600'
          )}>
            {data.market_sentiment.charAt(0).toUpperCase() + data.market_sentiment.slice(1)}
          </span>
        </div>
      </div>
    </div>
  )
}
