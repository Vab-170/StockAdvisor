'use client'

import { useQuery } from '@tanstack/react-query'
import { StockCard } from '../components/StockCard'
import { MarketOverview } from '../components/MarketOverview'
import { LoadingSpinner } from '../components/Loading'
import { ErrorMessage } from '../components/ErrorMessage'
import { getMarketSummary } from '../lib/api'

export default function HomePage() {
  const { data: marketData, isLoading, error } = useQuery({
    queryKey: ['market-summary'],
    queryFn: getMarketSummary,
    refetchInterval: 60000, // Refetch every minute
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ErrorMessage 
          message="Failed to load market data" 
          onRetry={() => window.location.reload()} 
        />
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Market Overview
        </h1>
        <p className="text-gray-600">
          Real-time AI-powered stock predictions and market analysis
        </p>
      </div>

      {marketData && (
        <>
          <MarketOverview data={marketData} />
          
          <div className="mt-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Stock Predictions
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {marketData.predictions.map((prediction) => (
                <StockCard
                  key={prediction.ticker}
                  prediction={prediction}
                />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
