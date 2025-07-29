#!/bin/bash

# StockAdvisor Deployment Script for Vercel

echo "🚀 Starting StockAdvisor Deployment to Vercel"

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "📁 Deploying Backend..."
cd backend
vercel --prod
BACKEND_URL=$(vercel ls stockadvisor-backend --token $VERCEL_TOKEN | grep "https" | head -1 | awk '{print $2}')
echo "✅ Backend deployed to: $BACKEND_URL"

echo "📁 Deploying Frontend..."
cd ../frontend

# Update environment variable with backend URL
echo "NEXT_PUBLIC_API_URL=$BACKEND_URL" > .env.production

vercel --prod
FRONTEND_URL=$(vercel ls stockadvisor-frontend --token $VERCEL_TOKEN | grep "https" | head -1 | awk '{print $2}')

echo "🎉 Deployment Complete!"
echo "📊 Frontend: $FRONTEND_URL"
echo "🔧 Backend: $BACKEND_URL"

echo "🔧 Don't forget to:"
echo "  1. Set OPENAI_API_KEY in Vercel backend environment variables"
echo "  2. Update CORS_ORIGINS in backend to include: $FRONTEND_URL"
