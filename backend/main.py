from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Stock Advisor API", description="Real-time stock market data API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Stock Advisor API is running!", "version": "1.0.0"}

@app.get("/api/stocks")
def get_stocks():
    return {
        "message": "Stock data retrieved successfully",
        "data": [
            {"symbol": "AAPL", "price": 150.25, "change": "+2.5%", "volume": "45.2M"},
            {"symbol": "GOOGL", "price": 2750.80, "change": "-1.2%", "volume": "28.1M"},
            {"symbol": "TSLA", "price": 850.45, "change": "+5.8%", "volume": "67.8M"},
            {"symbol": "MSFT", "price": 310.15, "change": "+1.8%", "volume": "32.4M"},
            {"symbol": "AMZN", "price": 3180.90, "change": "-0.5%", "volume": "25.7M"}
        ],
        "timestamp": "2025-08-20T12:00:00Z",
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
