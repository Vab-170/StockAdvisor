# StockAdvisor Deployment Script for Vercel (PowerShell)

Write-Host "🚀 Starting StockAdvisor Deployment to Vercel" -ForegroundColor Green

# Check if vercel CLI is installed
if (-not (Get-Command vercel -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Vercel CLI not found. Installing..." -ForegroundColor Red
    npm install -g vercel
}

Write-Host "📁 Deploying Backend..." -ForegroundColor Blue
Set-Location backend
vercel --prod

Write-Host "📁 Deploying Frontend..." -ForegroundColor Blue
Set-Location ..\frontend
vercel --prod

Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "🔧 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Set OPENAI_API_KEY in Vercel backend environment variables"
Write-Host "  2. Update NEXT_PUBLIC_API_URL in frontend environment variables"
Write-Host "  3. Update CORS_ORIGINS in backend environment variables"
