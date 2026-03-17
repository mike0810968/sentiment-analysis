import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Optional
import yfinance as yf
import feedparser
from newsapi import NewsApiClient
import json

# Ensure src is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import list_available_companies


class StockDataUpdater:
    """Class to handle real-time stock data updates"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with optional API key for premium services

        For free tier, you can use:
        - Yahoo Finance (yfinance library) - Free, good for basic data
        - Alpha Vantage (limited free requests)
        - Polygon.io (free tier available)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_dir = os.path.join(ROOT_DIR, "data")

    def update_stock_prices(self, tickers: List[str] = None, days_back: int = 365) -> dict:
        """
        Update stock prices using Yahoo Finance (free and reliable)

        Args:
            tickers: List of stock tickers to update. If None, updates all available
            days_back: Number of days of historical data to fetch

        Returns:
            Dict with update status for each ticker
        """
        if tickers is None:
            tickers = list_available_companies()

        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for ticker in tickers:
            try:
                print(f"Updating {ticker}...")

                # Fetch data from Yahoo Finance
                stock = yf.Ticker(f"{ticker}.NS")  # .NS for NSE stocks
                df = stock.history(start=start_date, end=end_date, interval='1d')

                if not df.empty:
                    # Format columns to match existing data structure
                    df = df.reset_index()
                    df = df.rename(columns={
                        'Date': 'Date',
                        'Open': 'Open',
                        'High': 'High',
                        'Low': 'Low',
                        'Close': 'Close',
                        'Volume': 'Volume'
                    })
                    df['Date'] = df['Date'].dt.date
                    df['Symbol'] = ticker

                    # Save to CSV (update existing or create new)
                    filename = f"{ticker}.csv"
                    filepath = os.path.join(self.base_dir, filename)

                    # Check if file exists and append new data
                    if os.path.exists(filepath):
                        existing_df = pd.read_csv(filepath)
                        existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.date

                        # Combine and remove duplicates
                        combined_df = pd.concat([existing_df, df])
                        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
                        combined_df = combined_df.sort_values('Date')
                    else:
                        combined_df = df

                    combined_df.to_csv(filepath, index=False)
                    results[ticker] = f"Updated {len(df)} days of data"
                    print(f"✅ {ticker}: Updated successfully")

                else:
                    results[ticker] = "No data available"
                    print(f"❌ {ticker}: No data available")

                # Rate limiting to avoid being blocked
                time.sleep(1)

            except Exception as e:
                results[ticker] = f"Error: {str(e)}"
                print(f"❌ {ticker}: Error - {str(e)}")

        return results

    def update_with_alpha_vantage(self, tickers: List[str] = None) -> dict:
        """
        Alternative method using Alpha Vantage API (requires API key)
        More reliable for intraday data but has rate limits
        """
        if not self.api_key:
            return {"error": "Alpha Vantage API key required"}

        results = {}

        for ticker in tickers[:5]:  # Limited due to API rate limits
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}.NS&apikey={self.api_key}&outputsize=full"
                response = requests.get(url)
                data = response.json()

                if "Time Series (Daily)" in data:
                    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                    df = df.rename(columns={
                        '1. open': 'Open',
                        '2. high': 'High',
                        '3. low': 'Low',
                        '4. close': 'Close',
                        '5. volume': 'Volume'
                    })
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    df['Symbol'] = ticker

                    filepath = os.path.join(self.base_dir, f"{ticker}.csv")
                    df.to_csv(filepath)
                    results[ticker] = "Updated with Alpha Vantage"
                else:
                    results[ticker] = "API limit reached or invalid response"

                time.sleep(12)  # Alpha Vantage rate limit: 5 calls/minute

            except Exception as e:
                results[ticker] = f"Error: {str(e)}"

        return results


class NewsDataUpdater:
    """Class to handle real-time news data updates"""

    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize with NewsAPI key for comprehensive news coverage

        Free tier: 100 requests/day
        Get API key from: https://newsapi.org/
        """
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.news_file = os.path.join(ROOT_DIR, "economictimes_data.csv")

    def update_news_from_newsapi(self, days_back: int = 7) -> dict:
        """
        Fetch latest financial news using NewsAPI
        """
        if not self.news_api_key:
            return {"error": "NewsAPI key required. Get from https://newsapi.org/"}

        try:
            newsapi = NewsApiClient(api_key=self.news_api_key)

            # Get news from last week
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # Search for Indian market news
            all_articles = []

            # Multiple queries for comprehensive coverage
            queries = [
                'Indian stock market OR NSE OR BSE',
                'company earnings OR financial results India',
                'economic news India',
                'corporate news India'
            ]

            for query in queries:
                try:
                    articles = newsapi.get_everything(
                        q=query,
                        from_param=from_date,
                        language='en',
                        sort_by='publishedAt',
                        page_size=50
                    )

                    for article in articles.get('articles', []):
                        all_articles.append({
                            'title': article['title'],
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article['url'],
                            'publishedAt': article['publishedAt'],
                            'source': article['source']['name']
                        })

                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    print(f"Error fetching news for query '{query}': {e}")

            # Convert to DataFrame
            if all_articles:
                news_df = pd.DataFrame(all_articles)

                # Clean and format
                news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
                news_df['date'] = news_df['publishedAt'].dt.strftime('%b %d, %Y, %I:%M %p IST')

                # Load existing news and combine
                if os.path.exists(self.news_file):
                    existing_news = pd.read_csv(self.news_file)
                    combined_news = pd.concat([existing_news, news_df])
                    combined_news = combined_news.drop_duplicates(subset=['title', 'url'], keep='last')
                else:
                    combined_news = news_df

                # Save updated news
                combined_news.to_csv(self.news_file, index=False)

                return {
                    "status": "success",
                    "new_articles": len(news_df),
                    "total_articles": len(combined_news)
                }
            else:
                return {"status": "no_new_articles"}

        except Exception as e:
            return {"error": str(e)}

    def update_news_from_rss(self) -> dict:
        """
        Alternative: Fetch news from RSS feeds (free, no API key needed)
        """
        rss_feeds = [
            'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'https://www.business-standard.com/rss/home_page_rss.xml',
            'https://www.financialexpress.com/feed/',
            'https://www.livemint.com/rss/homepage'
        ]

        all_news = []

        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:10]:  # Limit to recent entries
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6])

                    if published_date:
                        date_str = published_date.strftime('%b %d, %Y, %I:%M %p IST')
                    else:
                        date_str = datetime.now().strftime('%b %d, %Y, %I:%M %p IST')

                    all_news.append({
                        'title': entry.title,
                        'description': getattr(entry, 'summary', ''),
                        'content': getattr(entry, 'content', [{}])[0].get('value', '') if hasattr(entry, 'content') else '',
                        'url': entry.link,
                        'publishedAt': published_date or datetime.now(),
                        'date': date_str,
                        'source': feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else 'RSS Feed'
                    })

            except Exception as e:
                print(f"Error fetching RSS feed {feed_url}: {e}")

        if all_news:
            news_df = pd.DataFrame(all_news)

            # Load existing news and combine
            if os.path.exists(self.news_file):
                existing_news = pd.read_csv(self.news_file)
                combined_news = pd.concat([existing_news, news_df])
                combined_news = combined_news.drop_duplicates(subset=['title', 'url'], keep='last')
            else:
                combined_news = news_df

            combined_news.to_csv(self.news_file, index=False)

            return {
                "status": "success",
                "new_articles": len(news_df),
                "total_articles": len(combined_news)
            }

        return {"status": "no_new_articles"}


def main():
    """Main function to update both stock prices and news"""

    print("🚀 Starting daily data update...")

    # Update stock prices
    print("\n📈 Updating stock prices...")
    stock_updater = StockDataUpdater()

    # Update a few key stocks first (you can modify this list)
    key_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR']
    stock_results = stock_updater.update_stock_prices(key_stocks)

    print("Stock update results:")
    for ticker, result in stock_results.items():
        print(f"  {ticker}: {result}")

    # Update news
    print("\n📰 Updating news...")
    news_updater = NewsDataUpdater()

    # Try NewsAPI first, fallback to RSS
    news_results = news_updater.update_news_from_newsapi()

    if news_results.get('status') == 'error':
        print("Falling back to RSS feeds...")
        news_results = news_updater.update_news_from_rss()

    print("News update results:")
    print(f"  Status: {news_results.get('status', 'unknown')}")
    if 'new_articles' in news_results:
        print(f"  New articles: {news_results['new_articles']}")
    if 'total_articles' in news_results:
        print(f"  Total articles: {news_results['total_articles']}")

    print("\n✅ Daily update completed!")


if __name__ == "__main__":
    main()