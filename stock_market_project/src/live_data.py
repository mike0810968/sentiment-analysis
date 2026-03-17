import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure src is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import add_technical_indicators


class LiveStockData:
    """Real-time stock data handler using Yahoo Finance"""

    def __init__(self, cache_duration: int = 60):
        """
        Initialize live stock data handler

        Args:
            cache_duration: Cache data for this many seconds
        """
        self.cache_duration = cache_duration
        self.cache = {}
        self.last_update = {}

    def get_live_data(self, ticker: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """
        Get live stock data with improved accuracy for selected timeframes

        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            DataFrame with OHLCV data for the requested timeframe
        """
        cache_key = f"{ticker}_{period}_{interval}"
        current_time = time.time()

        # For very short cache duration on live data to ensure freshness
        live_cache_duration = 30  # 30 seconds for live data

        # Check if we have cached data that's still fresh
        if (cache_key in self.cache and
            cache_key in self.last_update and
            current_time - self.last_update[cache_key] < live_cache_duration):
            return self.cache[cache_key].copy()

        try:
            # Ensure ticker has .NS suffix for NSE
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"

            stock = yf.Ticker(ticker)

            # Strategy: Try to get data that matches the requested timeframe as closely as possible
            df = pd.DataFrame()

            # First, try the exact period and interval requested
            try:
                df = stock.history(period=period, interval=interval)
            except Exception as e:
                print(f"Failed to get {period} {interval} data: {e}")

            # If no data, try alternative strategies based on the period
            if df.empty:
                if period in ['1d', '5d']:
                    # For short periods, try 1-day interval
                    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                        try:
                            df = stock.history(period=period, interval="1m")
                        except:
                            try:
                                df = stock.history(period=period, interval="5m")
                            except:
                                df = stock.history(period=period, interval="1h")

                elif period in ['1mo', '3mo', '6mo']:
                    # For medium periods, try 1-hour or daily intervals
                    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                        try:
                            df = stock.history(period=period, interval="1h")
                        except:
                            df = stock.history(period=period, interval="1d")

                # For longer periods, ensure we get daily data
                elif period in ['1y', '2y', '5y', '10y', 'ytd', 'max']:
                    try:
                        df = stock.history(period=period, interval="1d")
                    except:
                        # Fallback to 1 year if longer period fails
                        df = stock.history(period="1y", interval="1d")

            # If still no data, try to get some recent data
            if df.empty:
                try:
                    # Try to get at least some recent data
                    df = stock.history(period="5d", interval="1h")
                    if df.empty:
                        df = stock.history(period="1mo", interval="1d")
                except Exception as e:
                    print(f"All data fetching attempts failed: {e}")

            if not df.empty:
                # Format the data
                df = df.reset_index()
                # Rename Datetime column to Date for consistency
                if 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'Date'})
                df['Date'] = pd.to_datetime(df['Date'])

                # Yahoo Finance already returns data for the requested period,
                # so no additional timeframe filtering is needed

                # Ensure we have required columns
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = np.nan

                # Add technical indicators
                df = self.add_technical_indicators(df)

                # Cache the data with timestamp
                self.cache[cache_key] = df.copy()
                self.last_update[cache_key] = current_time

                return df.copy()

        except Exception as e:
            print(f"Error fetching live data for {ticker}: {e}")

        # Return cached data if available, even if expired
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        # Return empty DataFrame as last resort
        return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe using existing functions"""
        if df.empty or len(df) < 20:
            return df

        try:
            # Use the existing add_technical_indicators function
            return add_technical_indicators(df)
        except Exception as e:
            st.warning(f"Error calculating technical indicators: {e}")
            return df

    def get_current_price(self, ticker: str) -> Dict:
        """Get current stock price and basic info with improved accuracy"""
        try:
            # Ensure proper ticker format
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"

            stock = yf.Ticker(ticker)
            info = stock.info

            # Try to get current price from multiple sources
            current_price = (
                info.get('currentPrice') or
                info.get('regularMarketPrice') or
                info.get('previousClose') or
                0
            )

            # Get recent trading data for more accuracy
            try:
                recent_data = stock.history(period="1d", interval="1m")
                if not recent_data.empty:
                    latest_price = recent_data['Close'].iloc[-1]
                    if latest_price > 0:
                        current_price = latest_price
            except:
                pass  # Keep the info-based price if recent data fails

            return {
                'current_price': current_price,
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', info.get('previousClose', 0)),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'company_name': info.get('longName', info.get('shortName', ticker.replace('.NS', ''))),
                'currency': info.get('currency', 'INR'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            # Try alternative approach with basic history data
            try:
                if not ticker.endswith('.NS'):
                    ticker = f"{ticker}.NS"
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    return {
                        'current_price': latest['Close'],
                        'previous_close': latest['Close'],  # Approximation
                        'open': latest['Open'],
                        'day_high': latest['High'],
                        'day_low': latest['Low'],
                        'volume': latest['Volume'],
                        'market_cap': 0,
                        'pe_ratio': 0,
                        'company_name': ticker.replace('.NS', ''),
                        'currency': 'INR'
                    }
            except:
                pass

            return {
                'current_price': 0,
                'error': str(e)
            }


class LiveNewsData:
    """Real-time financial news handler"""

    def __init__(self, news_api_key: Optional[str] = None, cache_duration: int = 300):
        """
        Initialize live news data handler

        Args:
            news_api_key: NewsAPI key for comprehensive news
            cache_duration: Cache news for this many seconds
        """
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.cache_duration = cache_duration
        self.cache = None
        self.last_update = 0
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def get_live_news(self, query: str = "Indian stock market OR NSE OR BSE", limit: int = 20) -> pd.DataFrame:
        """
        Get live financial news with sentiment analysis

        Args:
            query: Search query for news
            limit: Maximum number of articles to return

        Returns:
            DataFrame with news articles and sentiment scores
        """
        current_time = time.time()

        # Check if we have cached data that's still fresh
        if (self.cache is not None and
            current_time - self.last_update < self.cache_duration):
            print(f"Returning cached news data ({len(self.cache)} articles)")
            return self.cache.copy()

        news_data = []

        try:
            if self.news_api_key:
                print(f"Fetching news from NewsAPI with query: {query}")
                # Use NewsAPI for comprehensive news
                news_data = self._fetch_from_newsapi(query, limit)
                print(f"Fetched {len(news_data)} articles from NewsAPI")
            else:
                print("No NewsAPI key found, using RSS feeds")
                # Fallback to RSS feeds; pass query so we can filter RSS entries
                news_data = self._fetch_from_rss(limit, query=query)

            # If no results for a company-specific / complex query, try simpler fallbacks
            if not news_data and query and isinstance(query, str) and query.strip():
                try:
                    simple_queries = self._generate_simple_queries(query)
                    for sq in simple_queries:
                        print(f"No results for original query, trying simpler query: {sq}")
                        news_data = self._fetch_from_rss(limit, query=sq)
                        if news_data:
                            break
                except Exception as e:
                    print(f"Error generating simple queries: {e}")

            # Last resort: fetch general market headlines
            if not news_data:
                print("No company-specific news found; fetching general market headlines as fallback")
                news_data = self._fetch_from_rss(limit, query="Indian stock market OR NSE OR BSE")

            if news_data:
                print(f"Processing {len(news_data)} news articles")
                # Convert to DataFrame
                df = pd.DataFrame(news_data)

                # Perform sentiment analysis
                df = self._analyze_sentiment(df)

                # If the original query looks like a simple company ticker/name,
                # ensure articles that contain that token are tagged with that ticker.
                try:
                    if query and isinstance(query, str):
                        from src.data_preprocessing import list_available_companies

                        tickers = list_available_companies()
                        # Generate simple tokens from the query and map to tickers
                        simple_tokens = self._generate_simple_queries(query)
                        token_to_ticker = {}
                        for tok in simple_tokens:
                            tok_up = tok.upper().replace('"', '').strip()
                            for t in tickers:
                                if not isinstance(t, str):
                                    continue
                                t_up = t.upper()
                                t_base = t_up.replace('.NS', '')
                                if tok_up == t_up or tok_up == t_base:
                                    token_to_ticker[tok_up] = t
                                    break

                        if token_to_ticker:
                            # Build combined text for matching
                            def combined_text(row):
                                parts = [str(row.get('title','') or ''), str(row.get('description','') or ''), str(row.get('content','') or '')]
                                return ' '.join([p for p in parts if p]).upper()

                            df['_combined_tmp'] = df.apply(combined_text, axis=1)
                            # Update related_companies column if token present in text
                            if 'related_companies' not in df.columns:
                                df['related_companies'] = [[] for _ in range(len(df))]

                            for tok_up, mapped_ticker in token_to_ticker.items():
                                for idx, row in df.iterrows():
                                    text_up = row.get('_combined_tmp','')
                                    if tok_up in text_up:
                                        lst = row.get('related_companies') or []
                                        if mapped_ticker not in lst:
                                            lst = list(lst) if isinstance(lst, list) else [lst]
                                            lst.append(mapped_ticker)
                                            df.at[idx, 'related_companies'] = lst

                            df = df.drop(columns=['_combined_tmp'], errors='ignore')
                except Exception as e:
                    print(f"Error applying query-based tagging: {e}")

                # Cache the data
                self.cache = df.copy()
                self.last_update = current_time

                print(f"Successfully processed and cached {len(df)} news articles")
                return df.copy()
            else:
                print("No news data fetched from any source")

        except Exception as e:
            print(f"Error fetching live news: {e}")
            import traceback
            traceback.print_exc()

        # Return cached data if available
        if self.cache is not None:
            print(f"Returning cached news data due to error ({len(self.cache)} articles)")
            return self.cache.copy()

        print("Returning empty DataFrame")
        return pd.DataFrame()

    def set_api_key(self, key: Optional[str]):
        """Set or clear the NewsAPI key at runtime and clear cache."""
        if key:
            self.news_api_key = key
        else:
            self.news_api_key = None
        # Clear cache so subsequent calls use the new key
        self.cache = None
        self.last_update = 0

    def _fetch_from_newsapi(self, query: str, limit: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': limit,
            'apiKey': self.news_api_key
        }
        news_data = []
        news_data = []

        try:
            # Use a short timeout so UI doesn't hang on slow network
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:
                print("NewsAPI: failed to decode JSON response")
                return []

            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    news_data.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', ''),
                        'news_type': 'live',
                    })

        except requests.RequestException as e:
            print(f"NewsAPI request failed: {e}")

        return news_data

        return news_data

    def _fetch_from_rss(self, limit: int, query: Optional[str] = None) -> List[Dict]:
        """Fetch news from RSS feeds as fallback"""
        try:
            import feedparser
        except ImportError:
            print("Warning: feedparser not installed, cannot fetch RSS news")
            return []

        rss_feeds = [
            'https://www.moneycontrol.com/rss/latestnews.xml',
            'https://www.livemint.com/rss/markets',
            'https://www.thehindubusinessline.com/markets/?service=rss',
            'https://economictimes.indiatimes.com/rssfeeds/2146843.cms'
        ]

        # If a query is provided, prepend a Google News RSS search for that query
        # to get more real-time, query-specific results.
        if query and isinstance(query, str) and query.strip():
            try:
                import urllib.parse
                q = urllib.parse.quote_plus(query + " stock market NSE BSE")
                google_rss = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
                rss_feeds.insert(0, google_rss)
            except Exception:
                pass

        news_data = []

        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SentimentDashboard/1.0; +https://example.com)'
        }

        for idx, feed_url in enumerate(rss_feeds):
            try:
                print(f"Fetching RSS feed: {feed_url}")
                # Fetch with requests (with timeout) and hand content to feedparser
                try:
                    resp = requests.get(feed_url, timeout=10, headers=headers)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.content)
                except requests.RequestException as re:
                    print(f"Error fetching feed URL {feed_url}: {re}")
                    continue

                if not getattr(feed, 'entries', None):
                    print(f"No entries found in feed: {feed_url}")
                    continue

                # Determine how many entries to take from this feed.
                if query and idx == 0:
                    entries_to_take = min(len(feed.entries), max(limit, limit * 2))
                else:
                    entries_to_take = max(1, limit // max(1, len(rss_feeds)))

                for entry in feed.entries[:entries_to_take]:
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])

                    news_data.append({
                        'title': getattr(entry, 'title', ''),
                        'description': getattr(entry, 'summary', ''),
                        'content': getattr(entry, 'content', [{}])[0].get('value', '') if hasattr(entry, 'content') else '',
                        'url': getattr(entry, 'link', ''),
                        'source': getattr(feed.feed, 'title', 'RSS Feed') if getattr(feed, 'feed', None) else 'RSS Feed',
                        'published_at': published_date.isoformat() if published_date else datetime.now().isoformat(),
                        'author': getattr(entry, 'author', ''),
                        'news_type': 'live',
                    })

                print(f"Successfully fetched {len(feed.entries[:entries_to_take])} entries from {feed_url}")

            except Exception as e:
                print(f"Error processing RSS feed {feed_url}: {e}")
                continue

        print(f"Total news articles fetched from RSS: {len(news_data)}")
        # If a query is provided, we have already prepended Google News search
        # which returns query-targeted results. To maximize real-time coverage,
        # return the collected news items without additional strict filtering
        # (we rely on the Google News search + token presence to provide relevancy).
        if query:
            # Deduplicate by URL
            seen = set()
            deduped = []
            for a in news_data:
                url = a.get('url') or a.get('link') or ''
                if url in seen:
                    continue
                seen.add(url)
                deduped.append(a)
            return deduped

        # No query: return all collected RSS items
        return news_data

    def _generate_simple_queries(self, query: str) -> List[str]:
        """Generate simpler fallback queries from a complex/or-style query.

        Examples:
        - '"RELIANCE INDUSTRIES" OR RELIANCE' -> ['RELIANCE INDUSTRIES', 'RELIANCE INDUSTRIES stock', 'RELIANCE INDUSTRIES.NS', 'RELIANCE']
        - removes tokens like 'stock market', 'NSE', 'BSE' and splits on OR
        """
        import re

        if not query or not isinstance(query, str):
            return []

        # Clean quotes/parentheses and normalize
        q = re.sub(r'["\(\)]', ' ', query)

        # Split on OR (case-insensitive)
        parts = [p.strip() for p in re.split(r'\bOR\b', q, flags=re.IGNORECASE) if p.strip()]

        tokens = []
        for p in parts:
            # Remove common noise words that reduce recall
            cleaned = re.sub(r'\b(stock market|NSE|BSE|financial|news|company|updates|earnings|results)\b', ' ', p, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                tokens.append(cleaned)

        queries = []
        for t in tokens:
            # Keep the cleaned token as-is
            queries.append(t)
            # Add a lightweight variant (search for stock or market)
            queries.append(f"{t} stock")
            # Add .NS variation for NSE tickers
            if not t.upper().endswith('.NS'):
                queries.append(f"{t}.NS")
            # If token is a single uppercase token (likely ticker), add plain form too
            if len(t.split()) == 1 and t.upper() == t:
                queries.append(t)

        # Deduplicate while preserving order
        seen = set()
        out = []
        for item in queries:
            if item not in seen:
                seen.add(item)
                out.append(item)

        return out

    def _analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis on news headlines"""
        if df.empty:
            return df

        # Reuse the centralized analyzer in src.sentiment_analysis for consistency
        try:
            from src.sentiment_analysis import analyze_sentiment
            from src.data_preprocessing import list_available_companies

            tickers = list_available_companies()
            # Use the title column for text_column parameter, analyze_sentiment will combine fields
            analyzed = analyze_sentiment(df, text_column='title', tickers=tickers)
            return analyzed
        except Exception as e:
            print(f"Falling back to local sentiment analyzer due to: {e}")
            # Fallback to the previous simple analyzer
            sentiments = []

            for _, row in df.iterrows():
                text = f"{row.get('title','')} {row.get('description','') or ''}"
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)

                # Classify sentiment
                compound = sentiment_scores['compound']
                if compound >= 0.05:
                    sentiment_label = 'positive'
                elif compound <= -0.05:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'

                sentiments.append({
                    'sentiment_score': compound,
                    'sentiment_label': sentiment_label,
                    'sentiment_pos': sentiment_scores['pos'],
                    'sentiment_neg': sentiment_scores['neg'],
                    'sentiment_neu': sentiment_scores['neu']
                })

            sentiment_df = pd.DataFrame(sentiments)
            df = pd.concat([df, sentiment_df], axis=1)

            return df


# Global instances for caching
live_stock_data = LiveStockData()
live_news_data = LiveNewsData()