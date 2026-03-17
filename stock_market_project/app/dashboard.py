import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import uuid

# Ensure project root (so `import src...` works) is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
# Add the project root to sys.path so Python can import the `src` package
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# Also keep SRC_DIR available in case some modules import directly from it
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.data_preprocessing import (
    filter_by_date,
    list_available_companies,
    list_nifty50_companies,
    load_nifty_index,
    load_stock_data,
)
from src.live_data import live_stock_data, live_news_data
from src.feature_engineering import add_technical_indicators, create_features
from src.model_training import (
    directional_accuracy,
    evaluate_regression,
    train_regression_models,
)
from src.sentiment_analysis import analyze_sentiment, load_news_data


TIME_RANGES = {
    "1 Day": ("1d", "1m"),
    "5 Days": ("5d", "5m"),
    "1 Month": ("1mo", "1h"),
    "3 Months": ("3mo", "1d"),
    "6 Months": ("6mo", "1d"),
    "1 Year": ("1y", "1d"),
    "2 Years": ("2y", "1d"),
    "5 Years": ("5y", "1d"),
}


def get_time_range_config(time_range_key: str):
    """Get period and interval for the selected time range"""
    return TIME_RANGES.get(time_range_key, ("1y", "1d"))


def render_price_chart(df: pd.DataFrame, ticker: str):
    """Create interactive live price chart with technical indicators"""
    if df.empty:
        st.warning("No data available for chart")
        return

    fig = go.Figure()

    # Candlestick chart if we have OHLC data
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
    else:
        # Line chart for close price
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#1f77b4", width=2),
            )
        )

    # Volume as bar chart
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(158,158,158,0.3)',
            yaxis='y2',
            opacity=0.3
        ))

    # Moving averages
    if "SMA_20" in df.columns and df["SMA_20"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["SMA_20"], mode="lines", name="SMA 20",
                line=dict(dash="dash", color="#ff9800")
            )
        )
    if "EMA_20" in df.columns and df["EMA_20"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["EMA_20"], mode="lines", name="EMA 20",
                line=dict(dash="dot", color="#9c27b0")
            )
        )

    # Bollinger Bands
    if "BBU_20_2.0" in df.columns and df["BBU_20_2.0"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["BBU_20_2.0"], mode="lines", name="BB Upper",
            line=dict(dash="dash", color="#2196f3", width=1)
        ))
    if "BBL_20_2.0" in df.columns and df["BBL_20_2.0"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["BBL_20_2.0"], mode="lines", name="BB Lower",
            line=dict(dash="dash", color="#2196f3", width=1)
        ))

    fig.update_layout(
        title=f"{ticker} - Live Price Chart",
        xaxis_title="Date/Time",
        yaxis_title="Price (INR)",
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, width='stretch')


def render_volume_chart(df: dict, ticker: str):
    fig = px.bar(
        df,
        x="Date",
        y="Volume",
        title=f"{ticker} - Volume",
        labels={"Volume": "Volume", "Date": "Date"},
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width='stretch')


def render_technical_indicators(df: pd.DataFrame):
    """Render comprehensive technical indicators charts"""
    st.subheader("Technical Indicators")

    if df.empty:
        st.warning("No data available for technical indicators")
        return

    # RSI Chart
    if "RSI_14" in df.columns and df["RSI_14"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI (14)", line=dict(color="#636efa")))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            yaxis_title="RSI",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig, width='stretch')

    # MACD Chart
    if all(col in df.columns for col in ["MACD_12_26_9", "MACDs_12_26_9"]) and df["MACD_12_26_9"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_12_26_9"], name="MACD", line=dict(color="#00ff00")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACDs_12_26_9"], name="Signal", line=dict(color="#ff0000")))
        if "MACDh_12_26_9" in df.columns:
            fig.add_trace(go.Bar(x=df["Date"], y=df["MACDh_12_26_9"], name="Histogram", marker_color='rgba(158,158,158,0.5)'))
        fig.update_layout(
            title="MACD (12, 26, 9)",
            yaxis_title="MACD",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig, width='stretch')

    # Bollinger Bands Chart
    if all(col in df.columns for col in ["BBL_20_2.0", "BBU_20_2.0", "Close"]) and df["Close"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BBL_20_2.0"], name="BB Lower", line=dict(dash="dash", color="#2196f3")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BBU_20_2.0"], name="BB Upper", line=dict(dash="dash", color="#2196f3")))
        if "BBM_20_2.0" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["BBM_20_2.0"], name="BB Middle", line=dict(dash="dot", color="#ff9800")))
        fig.update_layout(
            title="Bollinger Bands (20, 2)",
            yaxis_title="Price",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig, width='stretch')

    # Stochastic Oscillator
    if "STOCHk_14_3_3" in df.columns and df["STOCHk_14_3_3"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["STOCHk_14_3_3"], name="%K", line=dict(color="#636efa")))
        if "STOCHd_14_3_3" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["STOCHd_14_3_3"], name="%D", line=dict(color="#ef553b")))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(
            title="Stochastic Oscillator",
            yaxis_title="Stochastic",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig, width='stretch')

    # Williams %R
    if "WILLR_14" in df.columns and df["WILLR_14"].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["WILLR_14"], name="Williams %R", line=dict(color="#ab63fa")))
        fig.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=-80, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(
            title="Williams %R",
            yaxis_title="Williams %R",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig, width='stretch')


def render_nifty_index():
    st.subheader("NIFTY Index Overview")

    if not st.button("Load NIFTY index chart"):
        st.info("Click the button to load the NIFTY index proxy chart. This may take a few seconds.")
        return

    with st.spinner("Loading NIFTY index... this may take 10-20 seconds"):
        try:
            nifty = load_nifty_index()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=nifty["Date"],
                    y=nifty["NIFTY_INDEX"],
                    name="NIFTY Proxy",
                    line=dict(color="#ff7f0e"),
                )
            )
            fig.update_layout(
                title="NIFTY Proxy (Average Close of NIFTY 50)",
                xaxis_title="Date",
                yaxis_title="Index Level",
                hovermode="x unified",
                template="plotly_white",
            )
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.warning(f"Unable to load NIFTY index data: {e}")


def render_news_sentiment(selected_company: str, tickers: List[str]):
    """Render live financial news and sentiment analysis"""

    # Allow entering a NewsAPI key in the sidebar settings
    try:
        import streamlit as _st
        from src.live_data import live_news_data as _live_news
        with _st.sidebar.expander("News API (optional)"):
            _key = _st.text_input("NewsAPI key (paste here)", value=_st.session_state.get('newsapi_key',''), type='password', help="Optional: enable NewsAPI for broader coverage")
            if _st.button("Set NewsAPI key"):
                _live_news.set_api_key(_key.strip() or None)
                _st.session_state['newsapi_key'] = _key.strip()
                if _key.strip():
                    _st.success("NewsAPI key set — using NewsAPI for live news")
                else:
                    _st.info("NewsAPI key cleared — falling back to RSS/Google News")
    except Exception:
        # Sidebar input is optional; fail silently if import issues occur
        pass

    if selected_company == "All":
        # Market-wide news view
        st.subheader("📰 Market-Wide Financial News & Sentiment Analysis")

        st.markdown("""
        **Complete Market Overview**: Stay updated with the latest financial news and sentiment analysis across all major companies.
        Get a comprehensive view of market sentiment and trending topics.
        """)

        # Get broad market news
        with st.spinner("Fetching latest market-wide financial news..."):
            # Use a smaller limit to reduce initial load time
            news_df = live_news_data.get_live_news(query="Indian stock market NSE BSE financial news economy", limit=12)

        show_company_specific = False
    else:
        # Company-specific news view
        st.subheader(f"📰 {selected_company} News & Sentiment Analysis")

        st.markdown(f"""
        **Company-Specific Analysis**: Latest news and sentiment analysis for {selected_company}.
        Stay informed about company developments, analyst ratings, and market reactions.
        """)

        # Allow custom keyword input for the company
        custom_keyword = st.text_input(
            "Search company or keyword",
            value="",
            placeholder=f"e.g. {selected_company} earnings, {selected_company} acquisition",
            key="custom_company_search",
        )

        # Determine the query string for live news
        if custom_keyword.strip():
            # Use the custom keyword for more targeted search
            query_text = f"{custom_keyword.strip()} stock market NSE BSE financial news"
        else:
            # Try to build a user-friendly company name for search using the
            # curated `company_mappings.json`. If not available, fall back to
            # using the `live_stock_data` company name, then to the ticker.
            company_search_name = None
            try:
                mapping_path = os.path.join(SRC_DIR, "company_mappings.json")
                if os.path.isfile(mapping_path):
                    import json
                    with open(mapping_path, "r", encoding="utf-8") as fh:
                        mappings = json.load(fh)
                    vals = mappings.get(selected_company)
                    if vals and isinstance(vals, list) and vals:
                        company_search_name = vals[0]
            except Exception:
                company_search_name = None

            if not company_search_name:
                try:
                    info = live_stock_data.get_current_price(selected_company)
                    if isinstance(info, dict):
                        cname = info.get('company_name')
                        if cname and isinstance(cname, str) and len(cname) > 2:
                            company_search_name = cname
                except Exception:
                    company_search_name = None

            # Build a broad OR-style query combining mapping variants, ticker, and company name
            query_parts = []
            try:
                mapping_path = os.path.join(SRC_DIR, "company_mappings.json")
                if os.path.isfile(mapping_path):
                    import json
                    with open(mapping_path, "r", encoding="utf-8") as fh:
                        mappings = json.load(fh)
                    variants = mappings.get(selected_company, [])
                    for v in variants:
                        if isinstance(v, str) and v.strip():
                            query_parts.append(v.strip())
            except Exception:
                pass

            # include the company_search_name (from yfinance) and the ticker forms
            if company_search_name and company_search_name not in query_parts:
                query_parts.insert(0, company_search_name)
            # include ticker and ticker.NS
            if selected_company not in query_parts:
                query_parts.append(selected_company)
            if (selected_company + ".NS") not in query_parts:
                query_parts.append(selected_company + ".NS")

            # Create OR query joined by ' OR ' and include context keywords
            or_query = " OR ".join([f'"{p}"' if ' ' in p else p for p in query_parts])
            query_text = f"{or_query} stock market NSE BSE financial news company updates earnings results"

        # Get company-specific news
        with st.spinner(f"Fetching latest news for {selected_company}..."):
            # Limit company-specific fetch to speed up responses
            news_df = live_news_data.get_live_news(query=query_text, limit=10)

        # Also show historical news from the Economic Times dataset (for reference)
        try:
            historic_df = load_news_data()
            if not historic_df.empty and custom_keyword.strip():
                keyword_lower = custom_keyword.strip().lower()
                historic_df = historic_df[historic_df.apply(
                    lambda row: keyword_lower in str(row.get('title', '')).lower() or keyword_lower in str(row.get('intro', '')).lower(),
                    axis=1,
                )]
                if not historic_df.empty:
                    # Ensure column names align with live news schema
                    historic_df = historic_df.rename(columns={
                        'date': 'published_at',
                        'intro': 'description',
                        'href': 'url',
                    })

                    # Provide missing fields expected by the dashboard
                    if 'title' not in historic_df.columns:
                        historic_df['title'] = ''
                    if 'description' not in historic_df.columns:
                        historic_df['description'] = ''
                    if 'url' not in historic_df.columns:
                        historic_df['url'] = ''

                    historic_df['source'] = 'EconomicTimes Archive'
                    historic_df['news_type'] = 'historic'
                    news_df = pd.concat([news_df, historic_df], ignore_index=True, sort=False)
        except Exception as e:
            st.warning(f"Could not load historic news data: {e}")
            # Continue without historic data

        show_company_specific = True

    # Manual refresh button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh News", key="refresh_news"):
            st.rerun()
    with col2:
        st.caption("Last updated: Auto-refresh enabled")

    if news_df.empty:
        st.error("❌ Unable to fetch live news data. Please check your internet connection.")
        st.info("💡 **Troubleshooting**: The app uses RSS feeds as fallback. If you're seeing this error, there might be a temporary issue with the news sources.")
        return

    # Analyze sentiment for all companies
    sentiment_df = analyze_sentiment(news_df, text_column="title", tickers=tickers)

    # Filter by selected company if a specific company is selected
    if selected_company != "All":
        related_series = sentiment_df.get("related_companies")
        if related_series is None:
            related_series = pd.Series([[]] * len(sentiment_df), index=sentiment_df.index)

        company_filtered = sentiment_df[related_series.apply(
            lambda lst: isinstance(lst, list) and selected_company in lst
        )]

        if not company_filtered.empty:
            sentiment_df = company_filtered
            st.success(f"✅ Found {len(sentiment_df)} news articles specifically about {selected_company}")
        else:
            # If custom keyword was used, try keyword-based filtering instead of strict related_company matching
            if 'custom_keyword' in locals() and custom_keyword.strip():
                keyword_lower = custom_keyword.strip().lower()
                keyword_filtered = sentiment_df[sentiment_df.apply(
                    lambda row: (keyword_lower in str(row.get('title','')).lower()) or
                                (keyword_lower in str(row.get('description','')).lower()) or
                                (keyword_lower in str(row.get('content','')).lower()),
                    axis=1
                )]

                if not keyword_filtered.empty:
                    sentiment_df = keyword_filtered
                    st.success(f"✅ Found {len(sentiment_df)} news articles matching keyword '{custom_keyword.strip()}' for {selected_company}")
                else:
                    st.info(f"ℹ️ No recent news found specifically about {selected_company}. Showing broader market headlines instead.")
                    # Fallback: fetch general market headlines so the user still sees fresh news
                    with st.spinner("Fetching market-wide headlines as fallback..."):
                        fallback_news = live_news_data.get_live_news(query="Indian stock market NSE BSE financial news", limit=10)
                    if fallback_news is not None and not fallback_news.empty:
                        sentiment_df = fallback_news
                    else:
                        st.info(f"No general market headlines available right now.")
                        return
            else:
                # If no custom keyword, fetch market-wide headlines as fallback instead of returning empty
                st.info(f"ℹ️ No recent news found specifically about {selected_company}. Showing broader market headlines instead.")
                with st.spinner("Fetching market-wide headlines as fallback..."):
                    fallback_news = live_news_data.get_live_news(query="Indian stock market NSE BSE financial news", limit=10)
                if fallback_news is not None and not fallback_news.empty:
                    sentiment_df = fallback_news
                else:
                    st.info(f"No general market headlines available right now.")
                    return

    # Sort by publication date (newest first)
    sentiment_df = sentiment_df.sort_values("published_at", ascending=False)

    # News filter options
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            ["All", "Positive", "Neutral", "Negative"],
            default=["All"],
            key="sentiment_filter"
        )

    with col2:
        show_company_news = st.checkbox("Company-specific only", value=False, key="company_filter")

    with col3:
        time_filter = st.selectbox(
            "Time Period",
            ["All Time", "Last 24h", "Last 12h", "Last 6h"],
            index=0,
            key="time_filter"
        )

    with col4:
        news_type_filter = st.selectbox(
            "News Type",
            ["All", "Live only", "Historic only"],
            index=0,
            key="news_type_filter"
        )

    # Apply filters
    filtered_df = sentiment_df.copy()

    # Ensure we always have a Series to work with for related company lists
    related_series = filtered_df.get('related_companies')
    if related_series is None:
        related_series = pd.Series([[]] * len(filtered_df), index=filtered_df.index)

    # Time filter
    if time_filter != "All Time" and pd.notna(filtered_df.get('published_at')).any():
        now = pd.Timestamp.now()
        if time_filter == "Last 24h":
            cutoff = now - pd.Timedelta(hours=24)
        elif time_filter == "Last 12h":
            cutoff = now - pd.Timedelta(hours=12)
        elif time_filter == "Last 6h":
            cutoff = now - pd.Timedelta(hours=6)

        filtered_df['published_datetime'] = pd.to_datetime(filtered_df['published_at'], errors='coerce')
        filtered_df = filtered_df[filtered_df['published_datetime'] >= cutoff]

    if "All" not in sentiment_filter:
        if "Positive" in sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'positive']
        if "Neutral" in sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'neutral']
        if "Negative" in sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment_label'] == 'negative']

    # News type filter
    if news_type_filter == "Live only":
        filtered_df = filtered_df[filtered_df.get('news_type') == 'live']
    elif news_type_filter == "Historic only":
        filtered_df = filtered_df[filtered_df.get('news_type') == 'historic']

    if show_company_news:
        filtered_df = filtered_df[related_series.apply(lambda lst: isinstance(lst, list) and len(lst) > 0)]

    # Display news
    if selected_company == "All":
        st.markdown("### 🌍 Latest Market News & Sentiment")
        st.markdown("Comprehensive financial news feed covering all major companies and market developments.")
    else:
        st.markdown(f"### 🏢 {selected_company} Latest News & Sentiment")
        st.markdown(f"Dedicated news feed for {selected_company} with real-time sentiment analysis.")

    if filtered_df.empty:
        if selected_company == "All":
            st.info("No news articles available at the moment. Please try refreshing or check back later.")
        else:
            st.info(f"No specific news found for {selected_company}. Try selecting 'All Companies' to see general market news.")
        return

    # Show news articles
    for _, row in filtered_df.head(20).iterrows():
        sentiment_color = {
            "positive": "🟢",
            "neutral": "🟡",
            "negative": "🔴"
        }.get(row.get("sentiment_label", "neutral"), "🟡")

        company_info = ""
        if selected_company == "All":
            related = row.get('related_companies') or []
            if isinstance(related, list) and related:
                company_info = f" **🏢 {', '.join(related[:3])}" + ("..." if len(related) > 3 else "") + "**"

        # Create expandable news item for better UX
        title_display = row['title'][:80] + ('...' if len(row['title']) > 80 else '')
        news_type_icon = "📰" if row.get('news_type') == 'live' else "📜"
        with st.expander(f"{sentiment_color} {news_type_icon} {title_display}{company_info}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Sentiment:** {row.get('sentiment_label', 'neutral').title()} ({row.get('sentiment_score', 0):.3f})")
                st.markdown(f"**Source:** {row.get('source', 'Unknown')}")
                st.markdown(f"**Published:** {row.get('published_at', 'Unknown') if pd.notna(row.get('published_at')) else 'Unknown'}")

                if pd.notna(row.get('description')) and row['description']:
                    st.markdown("**Summary:**")
                    st.write(row['description'])

            with col2:
                if pd.notna(row.get('url')) and row['url']:
                    st.markdown(f"[🔗 Read Full Article]({row['url']})")

                # Show sentiment gauge
                sentiment_score = row.get('sentiment_score', 0)
                if sentiment_score > 0.05:
                    st.success("Positive")
                elif sentiment_score < -0.05:
                    st.error("Negative")
                else:
                    st.warning("Neutral")

        st.markdown("---")

    # Display news items
    if not filtered_df.empty:
        st.markdown(f"### 📰 Latest News for {selected_company}")

        for idx, row in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{row['title']}**")
                    if pd.notna(row.get('description')) and row['description']:
                        st.write(row['description'][:200] + "..." if len(str(row['description'])) > 200 else row['description'])

                    # Show publication date
                    if pd.notna(row.get('published_at')):
                        published_date = pd.to_datetime(row['published_at'])
                        st.caption(f"Published: {published_date.strftime('%B %d, %Y at %I:%M %p')}")

                    # Show source
                    if pd.notna(row.get('source')):
                        st.caption(f"Source: {row['source']}")

                with col2:
                    # Show sentiment gauge
                    sentiment_score = row.get('sentiment_score', 0)
                    if sentiment_score > 0.05:
                        st.success("Positive")
                    elif sentiment_score < -0.05:
                        st.error("Negative")
                    else:
                        st.warning("Neutral")

                st.markdown("---")


            # --- Sentiment summary charts (added below existing news UI) ---
            try:
                # Use a stable per-run id to ensure keys are unique but consistent during a session
                if '_sentiment_run_id' not in st.session_state:
                    st.session_state['_sentiment_run_id'] = uuid.uuid4().hex
                run_id = st.session_state['_sentiment_run_id']

                sentiment_counts = filtered_df.get('sentiment_label') if not filtered_df.empty else pd.Series([], dtype=str)
                sentiment_counts = sentiment_counts.fillna('neutral')

                # Ensure consistent ordering
                order = ['positive', 'neutral', 'negative']
                vc = sentiment_counts.value_counts().reindex(order).fillna(0).astype(int)

                st.markdown("### 📊 Sentiment Summary")

                # Bar chart showing counts
                fig_bar = px.bar(
                    x=vc.index,
                    y=vc.values,
                    labels={'x': 'Sentiment', 'y': 'Number of Articles'},
                    title='News Sentiment Counts',
                    color=vc.index,
                    color_discrete_map={'positive': '#2ca02c', 'neutral': '#ffbf00', 'negative': '#d62728'}
                )
                fig_bar.update_layout(showlegend=False, template='plotly_white')
                st.plotly_chart(fig_bar, use_container_width=True, key=f"sentiment_bar_{run_id}")

                # Donut chart showing distribution
                fig_pie = px.pie(
                    names=vc.index,
                    values=vc.values,
                    title='Sentiment Distribution',
                    color=vc.index,
                    color_discrete_map={'positive': '#2ca02c', 'neutral': '#ffbf00', 'negative': '#d62728'}
                )
                fig_pie.update_traces(hole=0.45, textinfo='percent+label')
                fig_pie.update_layout(template='plotly_white')
                st.plotly_chart(fig_pie, use_container_width=True, key=f"sentiment_pie_{run_id}")

                # Line chart showing sentiment counts over time (daily)
                try:
                    if 'published_at' in filtered_df.columns and pd.notna(filtered_df['published_at']).any():
                        times = pd.to_datetime(filtered_df['published_at'], errors='coerce')
                        tmp = filtered_df.copy()
                        tmp['published_dt'] = pd.to_datetime(tmp['published_at'], errors='coerce')
                        # Aggregate by date (day)
                        tmp['date_only'] = tmp['published_dt'].dt.floor('D')
                        if tmp['date_only'].notna().any():
                            ts = tmp.groupby(['date_only', 'sentiment_label']).size().reset_index(name='count')
                            # Pivot for plotting
                            pivot = ts.pivot(index='date_only', columns='sentiment_label', values='count').fillna(0).reindex(columns=order, fill_value=0)
                            # Create line chart with three traces
                            fig_line = go.Figure()
                            colors = {'positive': '#2ca02c', 'neutral': '#ffbf00', 'negative': '#d62728'}
                            for s in order:
                                if s in pivot.columns:
                                    fig_line.add_trace(go.Scatter(x=pivot.index, y=pivot[s], mode='lines+markers', name=s.title(), line=dict(color=colors.get(s)), hovertemplate='%{y}'))
                            fig_line.update_layout(title='Sentiment Over Time (daily)', xaxis_title='Date', yaxis_title='Number of Articles', template='plotly_white', height=350)
                            st.plotly_chart(fig_line, use_container_width=True, key='sentiment_line')
                except Exception as _e:
                    # Non-critical: continue if timeseries chart cannot be built
                    pass
            except Exception as e:
                st.warning(f"Unable to render sentiment summary charts: {e}")




def render_model_predictions(df: pd.DataFrame, ticker: str):
    """Render ML model training and predictions with XGBoost and LSTM"""
    st.subheader("🤖 Machine Learning Predictions")
    st.markdown(
        "Train Random Forest, XGBoost, and LSTM models on historical features to predict next day stock price."
    )

# Diagnostics: surface why the section might be empty or blocked
    try:
        st.info(f"Data points available: {len(df)} | Columns: {list(df.columns)}")
        if not df.empty:
            st.write(df.tail(3))
    except Exception:
        # non-fatal: ignore display errors
        pass

    if df.empty or len(df) < 50:
        st.warning("Not enough data to train models. Need at least 50 data points.")
        return

    # Create features and show diagnostics so user knows if feature creation failed
    features = create_features(df)
    try:
        st.info(f"Features shape: {features.shape}")
        if not features.empty:
            st.write(features.head(3))
    except Exception:
        pass

    if features.empty:
        st.warning("Could not create features for model training.")
        return

    if st.button("Train ML Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes"):
            try:
                models = train_regression_models(features, target_col="Close")

                metrics = []
                for name, info in models.items():
                    y_test = info["y_test"].values
                    y_pred = info["y_pred"]
                    evals = evaluate_regression(y_test, y_pred)
                    evals["directional_accuracy"] = directional_accuracy(y_test, y_pred)
                    evals["model"] = name.replace("_", " ").title()
                    metrics.append(evals)

                metrics_df = pd.DataFrame(metrics).set_index("model")
                st.subheader("Model Performance Metrics")
                st.dataframe(metrics_df.style.highlight_max(axis=0), width='stretch')

                # Plot actual vs predicted for best model (lowest RMSE)
                best_model_name = metrics_df['rmse'].idxmin()
                best_key = best_model_name.lower().replace(" ", "_")
                if best_key in models:
                    best = models[best_key]
                    results = pd.DataFrame(
                        {
                            "Date": best["X_test"].index,
                            "Actual": best["y_test"].values,
                            "Predicted": best["y_pred"],
                        }
                    )

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results["Date"], y=results["Actual"], name="Actual", line=dict(color="#1f77b4")))
                    fig.add_trace(go.Scatter(x=results["Date"], y=results["Predicted"], name="Predicted", line=dict(color="#ff7f0e")))
                    fig.update_layout(
                        title=f"{ticker}: {best_model_name} - Actual vs Predicted Close Price",
                        xaxis_title="Date",
                        yaxis_title="Price (INR)",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Next day prediction
                    st.subheader("Next Day Price Prediction")
                    try:
                        # Use the last available data point for prediction
                        last_features = features.iloc[-1:].drop(columns=["Close"], errors="ignore")
                        last_features_scaled = best["scaler"].transform(last_features)

                        next_day_pred = best["model"].predict(last_features_scaled)[0]
                        current_price = df["Close"].iloc[-1] if not df.empty else 0

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"₹{current_price:.2f}")
                        with col2:
                            st.metric("Predicted Next Day", f"₹{next_day_pred:.2f}")
                        with col3:
                            pred_change = ((next_day_pred - current_price) / current_price) * 100
                            st.metric("Predicted Change", f"{pred_change:+.2f}%",
                                    delta=f"{pred_change:+.2f}%" if abs(pred_change) > 0.1 else "0.00%")

                    except Exception as e:
                        st.warning(f"Could not generate next day prediction: {e}")

            except Exception as e:
                st.error(f"Error training models: {e}")
                st.exception(e)


def render_stock_comparison(all_tickers: List[str]):
    st.markdown("Compare multiple stocks side-by-side to analyze performance, risk, and correlations.")

    # Stock selection
    col1, col2 = st.columns(2)
    with col1:
        num_stocks = st.selectbox("Number of stocks to compare", [2, 3, 4], index=1)

    # Company search and selection for comparison
    st.markdown("### Stock Selection")
    search_term = st.text_input("Search companies for comparison", placeholder="Type to search companies...", key="comparison_search")

    # Filter companies based on search term
    if search_term:
        filtered_tickers = [t for t in all_tickers if search_term.lower() in t.lower()]
    else:
        filtered_tickers = all_tickers

    # Show number of results
    if search_term:
        st.markdown(f"Found {len(filtered_tickers)} companies matching '{search_term}'")

    # Multi-select for stocks
    if filtered_tickers:
        selected_stocks = st.multiselect(
            f"Select {num_stocks} stocks to compare (scroll or search above)",
            filtered_tickers,
            default=filtered_tickers[:min(num_stocks, len(filtered_tickers))],
            max_selections=num_stocks,
            key="comparison_multiselect"
        )
    else:
        st.error("No companies found matching your search.")
        return

    if len(selected_stocks) < 2:
        st.warning("Please select at least 2 stocks to compare.")
        return

    # Time range selection
    time_range = st.selectbox("Time Range", list(TIME_RANGES.keys()), index=3)
    period, interval = get_time_range_config(time_range)

    # Load live data for selected stocks
    stock_data = {}
    for ticker in selected_stocks:
        try:
            with st.spinner(f"Loading {ticker}..."):
                df = live_stock_data.get_live_data(ticker, period=period, interval=interval)
            if not df.empty:
                stock_data[ticker] = df
        except Exception as e:
            st.warning(f"Could not load live data for {ticker}: {e}")

    if not stock_data:
        st.error("No data could be loaded for the selected stocks.")
        return

    # Normalized Price Comparison
    st.subheader("Normalized Price Comparison")
    st.markdown("All prices normalized to 100 at the start of the period for easy comparison.")

    fig = go.Figure()

    for ticker, df in stock_data.items():
        if not df.empty and len(df) > 0:
            # Normalize prices to start at 100
            start_price = df['Close'].iloc[0]
            normalized_prices = (df['Close'] / start_price) * 100

            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=normalized_prices,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))

    fig.update_layout(
        title="Normalized Price Performance",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base 100)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')

    # Performance Metrics Table
    st.subheader("Performance Metrics")

    metrics_data = []
    for ticker, df in stock_data.items():
        if not df.empty and len(df) > 1:
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            total_return = ((end_price - start_price) / start_price) * 100

            # Calculate volatility (standard deviation of returns)
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility

            # Calculate Sharpe ratio (assuming 5% risk-free rate)
            risk_free_rate = 0.05
            sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / (returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            metrics_data.append({
                'Stock': ticker,
                'Total Return (%)': round(total_return, 2),
                'Volatility (%)': round(volatility, 2),
                'Sharpe Ratio': round(sharpe_ratio, 2),
                'Max Drawdown (%)': round(max_drawdown, 2),
                'Start Price': round(start_price, 2),
                'End Price': round(end_price, 2)
            })

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, width='stretch')

        # Highlight best and worst performers
        best_return = metrics_df.loc[metrics_df['Total Return (%)'].idxmax()]
        worst_return = metrics_df.loc[metrics_df['Total Return (%)'].idxmin()]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 Best Performer: {best_return['Stock']} (+{best_return['Total Return (%)']}%)")
        with col2:
            st.error(f"📉 Worst Performer: {worst_return['Stock']} ({worst_return['Total Return (%)']}%)")

    # Correlation Matrix
    st.subheader("Price Correlation Matrix")
    st.markdown("Shows how closely stock prices move together. Values closer to 1 indicate strong positive correlation.")

    # Create correlation matrix
    price_data = pd.DataFrame()
    for ticker, df in stock_data.items():
        if not df.empty:
            price_data[ticker] = df.set_index('Date')['Close']

    # Align all series to common date range
    price_data = price_data.dropna()

    if not price_data.empty and len(price_data.columns) > 1:
        correlation_matrix = price_data.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title="Stock Price Correlation Matrix",
            template="plotly_white"
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Need at least 2 stocks with overlapping data to show correlation matrix.")

    # Risk-Return Scatter Plot
    st.subheader("Risk-Return Analysis")

    if metrics_data and len(metrics_data) > 1:
        risk_return_df = pd.DataFrame(metrics_data)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=risk_return_df['Volatility (%)'],
            y=risk_return_df['Total Return (%)'],
            mode='markers+text',
            text=risk_return_df['Stock'],
            textposition="top center",
            marker=dict(size=10, color='blue'),
            name='Stocks'
        ))

        fig.update_layout(
            title="Risk vs Return Scatter Plot",
            xaxis_title="Volatility (Risk) %",
            yaxis_title="Total Return %",
            template="plotly_white"
        )
        st.plotly_chart(fig, width='stretch')


def render_portfolio_management(all_tickers: List[str]):
    st.markdown("Create and manage your investment portfolio. Track performance, risk, and diversification.")

    # Initialize session state for portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}

    # Portfolio input section
    st.subheader("Portfolio Composition")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Add stock to portfolio
        st.markdown("**Add Stock to Portfolio**")

        # Search functionality for portfolio
        portfolio_search = st.text_input("Search companies for portfolio", placeholder="Type to search...", key="portfolio_search")

        # Filter companies based on search term
        if portfolio_search:
            filtered_portfolio_tickers = [t for t in all_tickers if portfolio_search.lower() in t.lower()]
        else:
            filtered_portfolio_tickers = all_tickers

        # Show number of results
        if portfolio_search:
            st.markdown(f"Found {len(filtered_portfolio_tickers)} companies")

        # Select stock from filtered list
        if filtered_portfolio_tickers:
            selected_stock = st.selectbox(
                "Select stock (scroll or search above)",
                filtered_portfolio_tickers,
                key="portfolio_stock"
            )
        else:
            st.error("No companies found matching your search.")
            selected_stock = None

        if selected_stock:
            investment_amount = st.number_input("Investment amount (₹)", min_value=1000, value=10000, step=1000)

            if st.button("Add to Portfolio"):
                if selected_stock in st.session_state.portfolio:
                    st.session_state.portfolio[selected_stock] += investment_amount
                else:
                    st.session_state.portfolio[selected_stock] = investment_amount
                st.success(f"Added ₹{investment_amount:,} to {selected_stock}")

    with col2:
        # Clear portfolio
        st.markdown("**Portfolio Actions**")
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = {}
            st.success("Portfolio cleared!")

        # Show current portfolio
        if st.session_state.portfolio:
            st.markdown("**Current Holdings**")
            portfolio_df = pd.DataFrame(
                list(st.session_state.portfolio.items()),
                columns=['Stock', 'Investment (₹)']
            )
            st.dataframe(portfolio_df, width='stretch')

    # Portfolio Analysis
    if st.session_state.portfolio:
        st.subheader("Portfolio Analysis")

        # Time range selection
        time_range = st.selectbox("Analysis Time Range", list(TIME_RANGES.keys()), index=3, key="portfolio_time")
        period, interval = get_time_range_config(time_range)

        # Load live data for portfolio stocks
        portfolio_data = {}
        total_investment = sum(st.session_state.portfolio.values())

        for stock, investment in st.session_state.portfolio.items():
            try:
                with st.spinner(f"Loading {stock}..."):
                    df = live_stock_data.get_live_data(stock, period=period, interval=interval)
                if not df.empty:
                    # Calculate weight
                    weight = investment / total_investment
                    portfolio_data[stock] = {
                        'data': df,
                        'weight': weight,
                        'investment': investment
                    }
            except Exception as e:
                st.warning(f"Could not load live data for {stock}: {e}")

        if portfolio_data:
            # Portfolio Performance Chart
            st.markdown("**Portfolio Performance**")

            # Create weighted portfolio returns
            portfolio_returns = pd.DataFrame()

            for stock, stock_info in portfolio_data.items():
                df = stock_info['data']
                weight = stock_info['weight']

                if not df.empty:
                    # Calculate daily returns
                    returns = df['Close'].pct_change() * weight
                    portfolio_returns[stock] = returns

            # Portfolio total returns
            portfolio_returns['Portfolio'] = portfolio_returns.sum(axis=1)
            portfolio_value = (1 + portfolio_returns['Portfolio']).cumprod() * total_investment

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=3)
            ))

            fig.update_layout(
                title=f"Portfolio Value Over Time (Initial Investment: ₹{total_investment:,})",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                template="plotly_white"
            )
            st.plotly_chart(fig, width='stretch')

            # Portfolio Metrics
            st.markdown("**Portfolio Metrics**")

            if len(portfolio_returns) > 1:
                # Calculate metrics
                total_return = (portfolio_value.iloc[-1] - total_investment) / total_investment * 100
                annualized_return = ((portfolio_value.iloc[-1] / total_investment) ** (252 / len(portfolio_returns)) - 1) * 100

                # Portfolio volatility
                portfolio_volatility = portfolio_returns['Portfolio'].std() * (252 ** 0.5) * 100

                # Sharpe ratio
                risk_free_rate = 0.05  # 5% annual risk-free rate
                sharpe_ratio = (portfolio_returns['Portfolio'].mean() * 252 - risk_free_rate) / (portfolio_returns['Portfolio'].std() * (252 ** 0.5))

                # Maximum drawdown
                cumulative = (1 + portfolio_returns['Portfolio']).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col2:
                    st.metric("Annualized Return", f"{annualized_return:.2f}%")
                with col3:
                    st.metric("Volatility", f"{portfolio_volatility:.2f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

            # Individual Stock Performance
            st.markdown("**Individual Stock Performance**")

            stock_performance = []
            for stock, stock_info in portfolio_data.items():
                df = stock_info['data']
                investment = stock_info['investment']

                if not df.empty and len(df) > 1:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    stock_return = ((end_price - start_price) / start_price) * 100
                    current_value = investment * (end_price / start_price)

                    stock_performance.append({
                        'Stock': stock,
                        'Investment': investment,
                        'Current Value': current_value,
                        'Return %': stock_return,
                        'Weight %': stock_info['weight'] * 100
                    })

            if stock_performance:
                perf_df = pd.DataFrame(stock_performance)
                st.dataframe(perf_df, width='stretch')

                # Asset Allocation Pie Chart
                st.markdown("**Asset Allocation**")

                fig = px.pie(
                    perf_df,
                    values='Weight %',
                    names='Stock',
                    title="Portfolio Allocation by Weight"
                )
                st.plotly_chart(fig, width='stretch')

        else:
            st.warning("No data available for portfolio analysis.")
    else:
        st.info("Add stocks to your portfolio to see analysis.")


def main():
    try:
        st.set_page_config(
            page_title="Live Stock Market Analysis & Prediction",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("📈 Live Stock Market Analysis and Prediction Platform")
        st.markdown(
            """Real-time stock analysis with live prices, interactive charts, financial news, and sentiment analysis."""
        )

        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            st.rerun()

        st.sidebar.header("Settings")

        # Add error handling for data loading
        try:
            tickers = list_available_companies()
            if not tickers:
                st.error("No stock data found. Please check your data directory.")
                return
        except Exception as e:
            st.error(f"Error loading company list: {e}")
            return

        # Prime the live news cache with a small fetch to reduce first-tab latency
        try:
            st.experimental_set_query_params()  # noop to ensure Streamlit context exists
            # Do a small, silent prefetch to warm caches (non-fatal)
            try:
                _ = live_news_data.get_live_news(limit=6)
            except Exception:
                pass
        except Exception:
            # If Streamlit context isn't ready to show messages, ignore
            pass

        # Company search and selection
        st.sidebar.markdown("### Company Selection")
        search_term = st.sidebar.text_input("Search companies", placeholder="Type to search...")

        # Filter companies based on search term
        if search_term:
            filtered_tickers = [t for t in tickers if search_term.lower() in t.lower()]
        else:
            filtered_tickers = tickers

        # Show number of results
        if search_term:
            st.sidebar.markdown(f"Found {len(filtered_tickers)} companies")

        # Select company from filtered list
        if filtered_tickers:
            # Add "All Companies" option for news
            news_options = ["All Companies"] + filtered_tickers
            default_index = 0  # Default to "All Companies" for news
            if "RELIANCE" in filtered_tickers:
                default_index = filtered_tickers.index("RELIANCE") + 1  # +1 because "All Companies" is first
            elif filtered_tickers and not search_term:  # Only set default if not searching
                default_index = (tickers.index("RELIANCE") + 1) if "RELIANCE" in tickers else 0

            selected_option = st.sidebar.selectbox(
                "Choose a company",
                news_options,
                index=min(default_index, len(news_options)-1),
                key="company_select"
            )

            # Set ticker for data loading (use first company if "All Companies" selected)
            if selected_option == "All Companies":
                ticker = filtered_tickers[0] if filtered_tickers else "RELIANCE"
                news_ticker = "All"
            else:
                ticker = selected_option
                news_ticker = selected_option

                # Show company info when specific company is selected
                try:
                    company_info = live_stock_data.get_current_price(ticker)
                    if company_info.get('current_price', 0) > 0:
                        st.sidebar.markdown("---")
                        st.sidebar.markdown(f"**{company_info.get('company_name', ticker)}**")
                        st.sidebar.metric("Current Price", f"₹{company_info['current_price']:.2f}")
                        if company_info.get('previous_close', 0) > 0:
                            change = company_info['current_price'] - company_info['previous_close']
                            change_pct = (change / company_info['previous_close']) * 100
                            st.sidebar.metric("Change", f"₹{change:.2f} ({change_pct:.2f}%)",
                                            delta=f"{change_pct:.2f}%" if change != 0 else "0.00%")
                except Exception as e:
                    st.sidebar.warning(f"Could not load company info: {e}")
        else:
            st.sidebar.error("No companies found matching your search.")
            return

        time_range = st.sidebar.selectbox("Time Range", list(TIME_RANGES.keys()), index=3)
        period, interval = get_time_range_config(time_range)

        st.sidebar.markdown("---")
        st.sidebar.markdown("#### About")
        st.sidebar.markdown(
            "Live stock market dashboard with real-time prices, interactive charts, **market-wide financial news**, and sentiment analysis. Data refreshes every 60 seconds."  # noqa: E501
        )

        # Load live data with error handling
        try:
            with st.spinner(f"Fetching live data for {ticker}..."):
                df = live_stock_data.get_live_data(ticker, period=period, interval=interval)
            if df.empty:
                st.error(f"No live data available for {ticker}")
                return
            # Technical indicators are already added in the live data fetch
        except Exception as e:
            st.error(f"Error loading live data for {ticker}: {e}")
            return

        tabs = st.tabs(["Overview", "Technical", "News & Sentiment", "Model Predictions", "Compare Stocks", "Portfolio"])

        with tabs[0]:
            st.header(f"📊 {ticker} Live Overview")
            st.markdown(
                "Real-time stock analysis with live prices, interactive charts, and technical indicators."
            )

            # Display current price information
            try:
                current_info = live_stock_data.get_current_price(ticker)
                if 'current_price' in current_info and current_info['current_price'] > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{current_info['current_price']:.2f}")
                    with col2:
                        change = current_info['current_price'] - current_info.get('previous_close', current_info['current_price'])
                        st.metric("Change", f"₹{change:.2f}", delta=f"{(change/current_info.get('previous_close', current_info['current_price'])*100):.2f}%")
                    with col3:
                        st.metric("Day High", f"₹{current_info.get('day_high', 0):.2f}")
                    with col4:
                        st.metric("Day Low", f"₹{current_info.get('day_low', 0):.2f}")
                else:
                    st.info("Current price information not available")
            except Exception as e:
                st.warning(f"Could not fetch current price: {e}")

            render_nifty_index()

            col1, col2 = st.columns(2)
            with col1:
                render_price_chart(df, ticker)
            with col2:
                render_volume_chart(df, ticker)

        with tabs[1]:
            st.header("Technical Analysis")
            render_technical_indicators(df)

        with tabs[2]:
            st.header("📰 Market News & Sentiment Analysis")
            render_news_sentiment(news_ticker, tickers)

        with tabs[3]:
            st.header("Model Training & Predictions")
            render_model_predictions(df, ticker)

        with tabs[4]:
            st.header("Multi-Stock Comparison")
            render_stock_comparison(tickers)

        with tabs[5]:
            st.header("Portfolio Management")
            render_portfolio_management(tickers)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
