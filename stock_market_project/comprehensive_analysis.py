#!/usr/bin/env python3
"""
Comprehensive Stock Market Analysis Script
Performs all dashboard analyses programmatically
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add src to path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import (
    filter_by_date,
    list_available_companies,
    list_nifty50_companies,
    load_nifty_index,
    load_stock_data,
)
from feature_engineering import add_technical_indicators, create_features
from model_training import (
    directional_accuracy,
    evaluate_regression,
    train_regression_models,
)
from sentiment_analysis import analyze_sentiment, load_news_data


def compare_stocks_side_by_side(tickers: List[str], days: int = 365):
    """Compare multiple stocks side by side"""
    print(f"\n{'='*60}")
    print("STOCK COMPARISON ANALYSIS")
    print(f"{'='*60}")

    comparison_data = {}

    for ticker in tickers:
        try:
            df = load_stock_data(ticker)
            df = add_technical_indicators(df)
            if days:
                max_date = df["Date"].max()
                min_date = max_date - pd.Timedelta(days=days)
                df = df[df["Date"] >= min_date]

            # Calculate key metrics
            returns = df["Close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

            comparison_data[ticker] = {
                "current_price": df["Close"].iloc[-1],
                "total_return_pct": total_return,
                "volatility": volatility,
                "avg_volume": df["Volume"].mean(),
                "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}"
            }

        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            continue

    # Display comparison table
    if comparison_data:
        print(f"\nStock Comparison (Last {days} days):")
        print("-" * 80)
        print(f"{'Ticker':<10} {'Current Price':<15} {'Total Return %':<15} {'Volatility':<12} {'Avg Volume':<12}")
        print("-" * 80)

        for ticker, data in comparison_data.items():
            print(f"{ticker:<10} {data['current_price']:<15.2f} {data['total_return_pct']:<15.2f} {data['volatility']:<12.4f} {data['avg_volume']:<12.0f}")

    return comparison_data


def analyze_technical_signals(ticker: str, days: int = 365):
    """Analyze technical indicators for trading signals"""
    print(f"\n{'='*60}")
    print(f"TECHNICAL ANALYSIS FOR {ticker.upper()}")
    print(f"{'='*60}")

    try:
        df = load_stock_data(ticker)
        df = add_technical_indicators(df)
        if days:
            max_date = df["Date"].max()
            min_date = max_date - pd.Timedelta(days=days)
            df = df[df["Date"] >= min_date]

        # Get latest values
        latest = df.iloc[-1]

        print(f"\nLatest Technical Indicators (as of {latest['Date'].date()}):")
        print(f"Close Price: ₹{latest['Close']:.2f}")

        # RSI Analysis
        if "RSI_14" in df.columns:
            rsi = latest["RSI_14"]
            rsi_signal = "OVERBOUGHT (>70)" if rsi > 70 else "OVERSOLD (<30)" if rsi < 30 else "NEUTRAL"
            print(f"RSI (14): {rsi:.2f} - {rsi_signal}")

        # MACD Analysis
        if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
            macd = latest["MACD_12_26_9"]
            signal = latest["MACDs_12_26_9"]
            macd_signal = "BULLISH" if macd > signal else "BEARISH"
            print(f"MACD: {macd:.4f} vs Signal: {signal:.4f} - {macd_signal}")

        # Bollinger Bands
        if "BBL_20_2.0" in df.columns and "BBU_20_2.0" in df.columns:
            close = latest["Close"]
            lower = latest["BBL_20_2.0"]
            upper = latest["BBU_20_2.0"]
            bb_position = (close - lower) / (upper - lower) * 100
            bb_signal = "NEAR UPPER BAND" if bb_position > 80 else "NEAR LOWER BAND" if bb_position < 20 else "MIDDLE RANGE"
            print(f"Bollinger Band Position: {bb_position:.1f}% - {bb_signal}")

        # Moving Averages
        if "SMA_20" in df.columns and "EMA_20" in df.columns:
            close = latest["Close"]
            sma = latest["SMA_20"]
            ema = latest["EMA_20"]
            sma_signal = "ABOVE SMA" if close > sma else "BELOW SMA"
            ema_signal = "ABOVE EMA" if close > ema else "BELOW EMA"
            print(f"Price vs SMA(20): {sma_signal} | Price vs EMA(20): {ema_signal}")

        # Recent trend
        recent_prices = df["Close"].tail(20)
        trend = "UPTREND" if recent_prices.iloc[-1] > recent_prices.iloc[0] else "DOWNTREND"
        trend_strength = abs((recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) * 100)
        print(f"Recent Trend (20 days): {trend} ({trend_strength:.2f}%)")

    except Exception as e:
        print(f"Error in technical analysis: {e}")


def monitor_news_sentiment(tickers: List[str]):
    """Monitor news sentiment analysis"""
    print(f"\n{'='*60}")
    print("NEWS SENTIMENT ANALYSIS")
    print(f"{'='*60}")

    try:
        news = load_news_data()
        if news.empty:
            print("No news data available.")
            return

        sentiment = analyze_sentiment(news, text_column="title", tickers=tickers)

        # Overall sentiment distribution
        sentiment_counts = sentiment["sentiment_label"].value_counts()
        total_news = len(sentiment)

        print(f"\nOverall Sentiment Distribution ({total_news} articles):")
        for label, count in sentiment_counts.items():
            percentage = (count / total_news) * 100
            print(f"{label.capitalize()}: {count} ({percentage:.1f}%)")

        # Company-specific sentiment
        company_sentiment = sentiment.groupby("related_company")["sentiment_label"].value_counts().unstack().fillna(0)
        print(f"\nSentiment by Company:")
        print(company_sentiment)

        # Recent sentiment trend (last 30 days)
        recent_sentiment = sentiment[sentiment["date"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
        if not recent_sentiment.empty:
            recent_counts = recent_sentiment["sentiment_label"].value_counts()
            print(f"\nRecent Sentiment (last 30 days, {len(recent_sentiment)} articles):")
            for label, count in recent_counts.items():
                percentage = (count / len(recent_sentiment)) * 100
                print(f"{label.capitalize()}: {count} ({percentage:.1f}%)")

        # Top positive and negative headlines
        print(f"\nTop 5 Positive Headlines:")
        positive_news = sentiment[sentiment["sentiment_label"] == "positive"].nlargest(5, "sentiment_score")
        for _, row in positive_news.iterrows():
            print(f"• {row['title'][:80]}... (Score: {row['sentiment_score']:.3f})")

        print(f"\nTop 5 Negative Headlines:")
        negative_news = sentiment[sentiment["sentiment_label"] == "negative"].nsmallest(5, "sentiment_score")
        for _, row in negative_news.iterrows():
            print(f"• {row['title'][:80]}... (Score: {row['sentiment_score']:.3f})")

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")


def experiment_with_ml_models(ticker: str):
    """Train and evaluate ML models for stock prediction"""
    print(f"\n{'='*60}")
    print(f"MACHINE LEARNING PREDICTIONS FOR {ticker.upper()}")
    print(f"{'='*60}")

    try:
        df = load_stock_data(ticker)
        df = add_technical_indicators(df)

        features = create_features(df)
        if features.empty:
            print("Not enough data to train models.")
            return

        print(f"Training models on {len(features)} data points...")

        models = train_regression_models(features, target_col="Close")

        print(f"\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Directional Acc %':<15}")
        print("-" * 80)

        best_model = None
        best_score = -float('inf')

        for name, info in models.items():
            y_test = info["y_test"].values
            y_pred = info["y_pred"]
            evals = evaluate_regression(y_test, y_pred)
            dir_acc = directional_accuracy(y_test, y_pred)

            print(f"{name:<20} {evals['rmse']:<10.4f} {evals['mae']:<10.4f} {evals['r2']:<10.4f} {dir_acc:<15.1f}")

            if evals['r2'] > best_score:
                best_score = evals['r2']
                best_model = name

        print(f"\nBest performing model: {best_model} (R² = {best_score:.4f})")

        # Show prediction accuracy for best model
        if best_model:
            best_info = models[best_model]
            y_test = best_info["y_test"].values
            y_pred = best_info["y_pred"]

            # Calculate prediction accuracy bands
            errors = y_test - y_pred
            mean_error = errors.mean()
            std_error = errors.std()

            print(f"\nPrediction Accuracy Analysis for {best_model}:")
            print(f"Mean Prediction Error: ₹{mean_error:.2f}")
            print(f"Standard Deviation of Errors: ₹{std_error:.2f}")
            print(f"95% Confidence Interval: ₹{mean_error - 1.96*std_error:.2f} to ₹{mean_error + 1.96*std_error:.2f}")

            # Directional accuracy details
            correct_directions = 0
            total_predictions = len(y_test) - 1

            for i in range(1, len(y_test)):
                actual_change = y_test[i] - y_test[i-1]
                pred_change = y_pred[i] - y_pred[i-1]

                if (actual_change > 0 and pred_change > 0) or (actual_change < 0 and pred_change < 0):
                    correct_directions += 1

            direction_acc = (correct_directions / total_predictions) * 100
            print(f"Detailed Directional Accuracy: {direction_acc:.1f}% ({correct_directions}/{total_predictions} correct)")

    except Exception as e:
        print(f"Error in ML modeling: {e}")


def explore_historical_trends(ticker: str):
    """Explore historical trends and patterns"""
    print(f"\n{'='*60}")
    print(f"HISTORICAL TRENDS ANALYSIS FOR {ticker.upper()}")
    print(f"{'='*60}")

    try:
        df = load_stock_data(ticker)
        df = add_technical_indicators(df)

        # Overall statistics
        total_days = len(df)
        start_date = df["Date"].min()
        end_date = df["Date"].max()
        total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

        print(f"\nHistorical Overview:")
        print(f"Data Range: {start_date.date()} to {end_date.date()} ({total_days} trading days)")
        print(f"Starting Price: ₹{df['Close'].iloc[0]:.2f}")
        print(f"Ending Price: ₹{df['Close'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return:.2f}%")

        # Annual returns
        df['Year'] = df['Date'].dt.year
        annual_returns = df.groupby('Year')['Close'].agg(['first', 'last'])
        annual_returns['return_pct'] = (annual_returns['last'] / annual_returns['first'] - 1) * 100

        print(f"\nAnnual Returns:")
        for year, row in annual_returns.iterrows():
            print(f"{year}: {row['return_pct']:.2f}%")

        # Best and worst years
        best_year = annual_returns['return_pct'].idxmax()
        worst_year = annual_returns['return_pct'].idxmin()
        print(f"\nBest Year: {best_year} ({annual_returns.loc[best_year, 'return_pct']:.2f}%)")
        print(f"Worst Year: {worst_year} ({annual_returns.loc[worst_year, 'return_pct']:.2f}%)")

        # Monthly seasonality
        df['Month'] = df['Date'].dt.month
        monthly_returns = df.groupby('Month')['Close'].agg(['first', 'last'])
        monthly_returns['return_pct'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100

        print(f"\nAverage Monthly Returns (Seasonality):")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in range(1, 13):
            if month in monthly_returns.index:
                ret = monthly_returns.loc[month, 'return_pct']
                print(f"{month_names[month-1]}: {ret:.2f}%")

        # Volatility analysis
        returns = df["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        max_drawdown = ((df["Close"] / df["Close"].expanding().max()) - 1).min() * 100

        print(f"\nRisk Metrics:")
        print(f"Annualized Volatility: {volatility:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio (assuming 5% risk-free rate): {(returns.mean() * 252 - 0.05) / volatility:.2f}")

        # Volume analysis
        avg_volume = df["Volume"].mean()
        max_volume = df["Volume"].max()
        volume_trend = "INCREASING" if df["Volume"].tail(100).mean() > df["Volume"].head(100).mean() else "DECREASING"

        print(f"\nVolume Analysis:")
        print(f"Average Daily Volume: {avg_volume:,.0f}")
        print(f"Maximum Daily Volume: {max_volume:,.0f}")
        print(f"Volume Trend: {volume_trend}")

    except Exception as e:
        print(f"Error in historical analysis: {e}")


def main():
    """Run all analyses"""
    print("🚀 COMPREHENSIVE STOCK MARKET ANALYSIS")
    print("=" * 80)

    # Get available companies
    try:
        all_tickers = list_available_companies()
        if not all_tickers:
            print("No stock data found!")
            return
    except Exception as e:
        print(f"Error loading companies: {e}")
        return

    # Select sample companies for comparison
    sample_tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    available_sample = [t for t in sample_tickers if t in all_tickers]

    if len(available_sample) < 2:
        available_sample = all_tickers[:5]  # Fallback to first 5

    # Run all analyses
    compare_stocks_side_by_side(available_sample, days=365)

    # Analyze each stock individually
    for ticker in available_sample[:3]:  # Limit to 3 for brevity
        analyze_technical_signals(ticker, days=365)
        experiment_with_ml_models(ticker)
        explore_historical_trends(ticker)

    # News sentiment (once for all)
    monitor_news_sentiment(all_tickers)

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print("Use this information to inform your trading decisions!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()