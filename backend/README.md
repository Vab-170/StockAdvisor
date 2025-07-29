# StockAdvisor Backend API

FastAPI backend for stock prediction and analysis.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy ML models:
```bash
cp -r ../ml_model/models/* ./models/
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run development server:
```bash
uvicorn app.main:app --reload
```

5. For production (Vercel):
```bash
vercel --prod
```

## API Endpoints

- `GET /` - Health check
- `GET /api/predict/{ticker}` - Get prediction for a stock
- `GET /api/stocks` - Get list of supported stocks
- `GET /api/market-summary` - Get market overview
- `POST /api/analyze` - Analyze custom stock list

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API key for explanations
- `ENVIRONMENT` - development/production
