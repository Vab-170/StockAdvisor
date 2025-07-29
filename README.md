# Stock Advisor - AI-Powered Stock Prediction Platform

A full-stack application that provides AI-powered stock predictions using machine learning models trained on technical indicators, market context, and fundamental analysis.

## 🚀 Features

- **Advanced ML Predictions**: 62-feature Random Forest model with 69.9% average accuracy
- **Real-time Market Data**: Live data from Yahoo Finance, VIX, Treasury yields, and more
- **Modern UI**: Responsive React frontend with TailwindCSS
- **Fast API Backend**: RESTful API built with FastAPI
- **Production Ready**: Configured for Vercel deployment

## 📊 Model Performance

- **NVDA**: 73.6% accuracy
- **AMZN**: 72.5% accuracy  
- **GOOGL**: 71.4% accuracy
- **Average**: 69.9% accuracy across all tested stocks

## 🏗️ Architecture

```
StockAdvisor/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── prediction_service.py  # ML service layer
│   │   └── schemas.py        # Data models
│   ├── models/        # Trained ML models (.pkl files)
│   └── vercel.json    # Vercel deployment config
├── frontend/          # Next.js frontend
│   ├── app/           # App Router pages
│   ├── components/    # React components
│   ├── lib/           # Utilities and API client
│   └── styles/        # TailwindCSS styles
└── ml_model/          # ML training and development
    ├── train_model.py      # Model training script
    ├── daily_summary.py    # Daily analysis script
    └── models/             # Generated model files
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **scikit-learn**: Machine learning models
- **yfinance**: Real-time stock data
- **OpenAI**: AI-powered explanations
- **uvicorn**: ASGI server

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **TailwindCSS**: Utility-first styling
- **TanStack Query**: Data fetching and caching
- **Lucide React**: Modern icons

### ML Pipeline
- **Random Forest**: 200 estimators, max_depth=10
- **StandardScaler**: Feature normalization
- **62 Features**: Technical indicators + market context + fundamentals

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- OpenAI API key (optional, for explanations)

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   copy .env.example .env
   # Edit .env with your OpenAI API key
   ```

4. **Start the API server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   copy .env.local.example .env.local
   # Edit .env.local with your API URL
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📈 ML Model Training

To retrain the model with fresh data:

1. **Navigate to ML directory**
   ```bash
   cd ml_model
   ```

2. **Install ML dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Copy models to backend**
   ```bash
   copy models\*.pkl ..\backend\models\
   ```

## 🌐 Deployment

### Vercel Deployment

**Backend (API)**
1. Connect your repository to Vercel
2. Select the `backend` folder as the root directory
3. Set environment variables in Vercel dashboard
4. Deploy as a Serverless Function

**Frontend**
1. Connect your repository to Vercel
2. Select the `frontend` folder as the root directory
3. Set `NEXT_PUBLIC_API_URL` to your backend URL
4. Deploy as a static site

### Environment Variables

**Backend (.env)**
```
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=production
CORS_ORIGINS=["https://your-frontend.vercel.app"]
```

**Frontend (.env.local)**
```
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

## 📚 API Endpoints

### GET /api/market-summary
Returns market overview with predictions for popular stocks.

### GET /api/predict/{ticker}
Get prediction for a specific stock ticker.

### POST /api/analyze
Analyze multiple stocks at once.
```json
{
  "tickers": ["AAPL", "TSLA", "NVDA"],
  "include_explanation": true
}
```

## 🔧 Features

### Technical Indicators (34 features)
- Moving averages (SMA 10, 20, 50, 200)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility measures (Bollinger Bands, ATR)
- Volume analysis

### Market Context (10 features)
- VIX (volatility index)
- Treasury yields (10Y, 2Y)
- Dollar Index (DXY)
- Sector momentum (SPY, QQQ, IWM)

### Fundamental Analysis (13 features)
- P/E ratios
- Revenue growth
- Profit margins
- Debt ratios
- ROE, ROA

### Combined Signals (5 features)
- Technical-fundamental score
- Risk-adjusted momentum
- Market correlation
- Sector relative strength

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. Stock predictions are not guaranteed and should not be considered as financial advice. Always do your own research and consult with a financial advisor before making investment decisions.