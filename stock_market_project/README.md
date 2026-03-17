# Stock Market Analysis and Prediction Platform

A comprehensive Streamlit-based financial analytics dashboard for Indian stock markets with **real-time data updates**, live charts, and sentiment analysis.

## ✨ Features

### 🔴 LIVE FEATURES
- **Live Stock Prices**: Real-time NSE stock data via Yahoo Finance with improved accuracy
- **Live Interactive Charts**: Plotly charts with hover, zoom, and pan
- **Company-Specific News**: Filter news by selected company or view market-wide news
- **Live Financial News**: Latest news from multiple RSS sources with sentiment analysis
- **Auto/Manual Refresh**: Manual refresh button for live data updates
- **Real-time Sentiment Analysis**: VADER sentiment scoring on live news
- **Smart Company Search**: Search and select companies with live price display

### 📊 ANALYTICS FEATURES
- **Interactive Stock Charts**: Price and volume visualizations with technical indicators
- **Multi-Stock Comparison**: Compare up to 4 stocks side-by-side with performance metrics
- **Portfolio Management**: Create and track custom investment portfolios
- **Advanced Analytics**: Risk metrics, correlation analysis, Sharpe ratios

### 🤖 MACHINE LEARNING
- **ML Predictions**: Random Forest, XGBoost, and Linear Regression models
- **Next Day Price Prediction**: Forecast tomorrow's stock price
- **Model Performance Metrics**: RMSE, MAE, Directional Accuracy

### 📈 TECHNICAL INDICATORS
- **RSI (Relative Strength Index)**
- **MACD (Moving Average Convergence Divergence)**
- **Bollinger Bands**
- **Stochastic Oscillator**
- **Williams %R**
- **Moving Averages (SMA, EMA)**

## 🚀 Getting Started

### 1. Installation

```bash
# Clone or navigate to project directory
cd stock_market_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
python main.py
# or
streamlit run app/dashboard.py
```

## 🔄 Real-Time Data Updates

### Live Data Sources
- **Stock Prices**: Yahoo Finance (yfinance library)
- **Financial News**: NewsAPI (optional, fallback to RSS feeds)
- **Sentiment Analysis**: VADER (pre-trained model)

### Manual Refresh
Click the "🔄 Refresh Data" button in the sidebar to update all live data.

### API Keys (Optional)
For enhanced news coverage, add NewsAPI key:
```bash
# Create .env file
NEWS_API_KEY=your_newsapi_key_here
```

## 📊 Dashboard Layout

### Sidebar
- **Company Search**: Searchable dropdown for NSE stocks
- **Time Range**: 1D, 5D, 1M, 3M, 6M, 1Y, 2Y, 5Y
- **Refresh Button**: Manual data refresh

### Main Tabs
1. **Overview**: Live price metrics, interactive charts
2. **Technical**: RSI, MACD, Bollinger Bands, Stochastic
3. **News & Sentiment**: Live financial news with sentiment
4. **Model Predictions**: ML models and price forecasts
5. **Compare Stocks**: Multi-stock analysis
6. **Portfolio**: Investment portfolio management

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Visualization**: Plotly
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Live Data**: yfinance, requests
- **Sentiment Analysis**: VADER
- **News API**: NewsAPI (optional)
# Edit .env with your API keys
```

#### Setup Automated Updates
```bash
# Make script executable
chmod +x setup_automation.sh

# Run setup script
./setup_automation.sh
```

This will:
- Update stock prices daily at 6:00 AM (Monday-Friday)
- Fetch latest news articles
- Log all updates to `logs/daily_update.log`

### Option 2: Cloud Automation (GitHub Actions)

1. **Fork this repository** to your GitHub account

2. **Add API Keys as Secrets**:
   - Go to repository Settings → Secrets and variables → Actions
   - Add: `NEWS_API_KEY` and `ALPHA_VANTAGE_API_KEY`

3. **Enable GitHub Actions**:
   - The workflow will automatically run daily at 6:00 AM IST
   - Manual triggers available via GitHub Actions tab

### Manual Updates

Run updates manually anytime:

```bash
# Update all data
python src/data_updater.py

# Update specific stocks
python -c "
from src.data_updater import StockDataUpdater
updater = StockDataUpdater()
results = updater.update_stock_prices(['RELIANCE', 'TCS', 'INFY'])
print(results)
"
```

## 📊 Data Sources

### Stock Data
- **Primary**: Yahoo Finance (via `yfinance` library)
- **Alternative**: Alpha Vantage API
- **Coverage**: 1,942+ NSE stocks + NIFTY50

### News Data
- **Primary**: NewsAPI (global financial news)
- **Fallback**: RSS feeds from Indian financial publications
- **Sentiment**: VADER sentiment analysis

## 🏗️ Project Structure

```
stock_market_project/
├── app/
│   └── dashboard.py          # Main Streamlit dashboard
├── src/
│   ├── data_preprocessing.py # Data loading functions
│   ├── feature_engineering.py # Technical indicators
│   ├── model_training.py     # ML model training
│   ├── sentiment_analysis.py # News sentiment analysis
│   └── data_updater.py       # Real-time data updates ⭐ NEW
├── data/                     # Stock CSV files
├── logs/                     # Update logs ⭐ NEW
├── .github/workflows/        # GitHub Actions ⭐ NEW
├── .env.example             # API keys template ⭐ NEW
├── setup_automation.sh      # Local automation script ⭐ NEW
├── requirements.txt         # Python dependencies
└── README.md
```

## 🎯 Usage Guide

### Dashboard Tabs

1. **Overview**: Price charts, volume, and NIFTY index
2. **Technical**: RSI, MACD, Bollinger Bands, Moving Averages
3. **News & Sentiment**: Latest news with sentiment scores
4. **Model Predictions**: ML model training and forecasting
5. **Compare Stocks**: Multi-stock comparison with metrics
6. **Portfolio**: Custom portfolio creation and tracking

### Search Functionality

- **Main Selector**: Search companies in sidebar
- **Comparisons**: Search when selecting stocks to compare
- **Portfolio**: Search when adding stocks to portfolio
- **News Filter**: Search companies mentioned in news

## 🔧 Customization

### Adding New Stocks
```python
from src.data_updater import StockDataUpdater
updater = StockDataUpdater()
updater.update_stock_prices(['NEW_STOCK_TICKER'])
```

### Modifying Update Schedule
- **Local**: Edit `setup_automation.sh` and re-run
- **GitHub**: Modify `.github/workflows/daily-update.yml`

### Custom News Sources
Edit `src/data_updater.py` to add new RSS feeds or API endpoints.

## 📈 Performance Metrics

The dashboard calculates:
- Total Return & Annualized Return
- Volatility & Sharpe Ratio
- Maximum Drawdown
- Risk-adjusted performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify.

## ⚠️ Disclaimer

This dashboard is for educational and informational purposes only. Not financial advice. Always do your own research before making investment decisions.
