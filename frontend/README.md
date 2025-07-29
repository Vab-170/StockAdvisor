# StockAdvisor Frontend

Modern React frontend for the StockAdvisor application, built with Next.js 14, TypeScript, and TailwindCSS.

## 🚀 Features

- **Modern UI**: Clean, responsive design with TailwindCSS
- **Real-time Data**: Live stock predictions and market data
- **Interactive Components**: Stock cards, market overview, search functionality
- **Type Safety**: Full TypeScript integration
- **Performance**: Optimized with TanStack Query for data fetching and caching
- **Mobile Responsive**: Works seamlessly on all device sizes

## 🛠️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **Icons**: Lucide React
- **Data Fetching**: TanStack Query (React Query)
- **HTTP Client**: Axios
- **Build Tool**: Next.js built-in webpack

## 📋 Prerequisites

- Node.js 18 or higher
- npm package manager
- Backend API running (see [backend README](../backend/README.md))

## ⚙️ Installation

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
   # Copy the example file
   copy .env.local.example .env.local
   
   # Edit .env.local with your settings
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

## 🚀 Running the Application

### Development Mode
```bash
npm run dev
```

The application will start on `http://localhost:3000`

### Production Build
```bash
# Build the application
npm run build

# Start production server
npm start
```

### Type Checking
```bash
npm run type-check
```

### Linting
```bash
npm run lint
```

## 🏗️ Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   ├── providers.tsx      # React Query provider
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── StockCard.tsx      # Individual stock display
│   ├── MarketOverview.tsx # Market summary
│   ├── Loading.tsx        # Loading components
│   └── ErrorMessage.tsx   # Error handling
├── lib/                   # Utilities
│   ├── api.ts            # API client and types
│   └── utils.ts          # Helper functions
└── styles/               # Additional styles
```

## 🔗 API Integration

The frontend communicates with the backend API through the following endpoints:

### Market Summary
- **Endpoint**: `GET /api/market-summary`
- **Component**: `MarketOverview`
- **Purpose**: Display overall market sentiment and top stock predictions

### Stock Prediction
- **Endpoint**: `GET /api/predict/{ticker}`
- **Component**: `StockCard`
- **Purpose**: Show individual stock analysis and predictions

### Stock Search
- **Endpoint**: `GET /api/predict/{ticker}`
- **Component**: Search form in main page
- **Purpose**: Allow users to search for specific stocks

## 🎨 UI Components

### StockCard
Displays individual stock information including:
- Stock ticker and current price
- Prediction (Buy/Hold/Sell) with confidence
- Technical indicators (RSI, daily return, volume)
- AI-powered explanation

### MarketOverview
Shows market-wide information:
- S&P 500 performance
- VIX (volatility index)
- Treasury yields
- Dollar index (DXY)
- Overall market sentiment

### Loading & Error States
- Skeleton loading components
- Error messages with retry functionality
- Graceful fallbacks for API failures

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the frontend directory:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Analytics
NEXT_PUBLIC_GA_ID=your_google_analytics_id
```

### TailwindCSS

The application uses a custom TailwindCSS configuration with:
- Custom color palette for buy/sell/hold predictions
- Responsive breakpoints
- Component utilities for cards and buttons

## 📱 Responsive Design

The application is fully responsive with breakpoints:
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

Key responsive features:
- Adaptive grid layouts
- Mobile-optimized navigation
- Touch-friendly interactive elements

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Ensure backend is running
   curl http://localhost:8000/
   
   # Check environment variables
   echo $NEXT_PUBLIC_API_URL
   ```

2. **Build Errors**
   ```bash
   # Clear Next.js cache
   rm -rf .next
   npm run build
   ```

3. **TypeScript Errors**
   ```bash
   # Run type checking
   npm run type-check
   ```

4. **Styling Issues**
   ```bash
   # Regenerate TailwindCSS
   npm run dev
   ```

### Development Tips

- Use React Developer Tools for component debugging
- Check Network tab for API call issues
- Use TypeScript strict mode for better error catching
- Utilize React Query DevTools for cache inspection

## 🚀 Deployment

### Vercel (Recommended)

1. **Connect your repository to Vercel**
2. **Set environment variables**:
   - `NEXT_PUBLIC_API_URL=https://your-backend-url.vercel.app`
3. **Deploy**

### Other Platforms

The application can be deployed to any platform that supports Next.js:
- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

### Build Output

```bash
npm run build
```

Creates optimized production build in `.next/` directory.

## 🎯 Features Roadmap

- [ ] Real-time WebSocket updates
- [ ] Advanced charting with technical indicators
- [ ] Portfolio tracking
- [ ] Alerts and notifications
- [ ] Dark mode theme
- [ ] Export functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
