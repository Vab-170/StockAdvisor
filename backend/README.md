# StockAdvisor Backend

FastAPI-based backend service for AI-powered stock predictions using machine learning models.

## 🚀 Features

- **Machine Learning Predictions**: 62-feature Random Forest models with 69.9% average accuracy
- **Real-time Market Data**: Live data from Yahoo Finance, VIX, Treasury yields, and market indices
- **AI Explanations**: OpenAI-powered explanations for predictions (optional)
- **RESTful API**: FastAPI with automatic documentation
- **CORS Support**: Configured for frontend integration

## 📊 Supported Stocks

- **AAPL** (Apple Inc.)
- **NVDA** (NVIDIA Corporation)
- **AMZN** (Amazon.com Inc.)
- **GOOGL** (Alphabet Inc.)
- **TSLA** (Tesla Inc.)

## 🛠️ Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key (optional, for enhanced explanations)

## ⚙️ Installation

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # Copy the example file
   copy .env.example .env
   
   # Edit .env with your settings
   OPENAI_API_KEY=your_openai_api_key_here
   ENVIRONMENT=development
   ```

## 🚀 Running the Server

### Development Mode
```bash
# From the project root directory
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## 📚 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### Health Check
```http
GET /
```
Returns server status and version information.

### Market Summary
```http
GET /api/market-summary
```
Returns predictions for all supported stocks with market sentiment.

### Individual Stock Prediction
```http
GET /api/predict/{ticker}
```
Returns prediction for a specific stock ticker.

**Example:**
```bash
curl http://localhost:8000/api/predict/AAPL
```

### Bulk Analysis
```http
POST /api/analyze
```
Analyze multiple stocks at once.

**Request Body:**
```json
{
  "tickers": ["AAPL", "NVDA", "TSLA"],
  "include_explanation": true
}
```

### Supported Stocks
```http
GET /api/stocks
```
Returns list of all supported stock tickers.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# OpenAI API Key (optional, for AI explanations)
OPENAI_API_KEY=your_openai_api_key_here

# Environment setting
ENVIRONMENT=development

# CORS origins (for production)
CORS_ORIGINS=["https://your-frontend-domain.com"]
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're running from the project root
   cd /path/to/StockAdvisor
   python -m uvicorn backend.app.main:app --reload
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Model Files Not Found**
   - Ensure model files are in `backend/models/` directory
   - Check that all required `.pkl` files exist for each ticker

4. **OpenAI API Issues**
   - Verify your API key in the `.env` file
   - Check your OpenAI account has sufficient credits

## 🚀 Deployment

The backend is configured for Vercel deployment. Set environment variables in your deployment platform:
- `OPENAI_API_KEY`
- `ENVIRONMENT=production`
