#!/usr/bin/env python3
"""Test script for news functionality with custom keyword"""

from src.live_data import live_news_data
from src.sentiment_analysis import load_news_data, analyze_sentiment
from src.data_preprocessing import list_available_companies
import pandas as pd

try:
    # Simulate typing 'Reliance' as custom keyword
    custom_keyword = 'Reliance'
    query_text = custom_keyword.strip()

    print(f"Testing with custom keyword: '{custom_keyword}'")

    # Get live news
    news_df = live_news_data.get_live_news(query=query_text, limit=5)
    print(f'Live news fetched: {len(news_df)} articles')

    # Load historic data
    historic_df = load_news_data()
    print(f'Historic data loaded: {len(historic_df)} articles')

    if not historic_df.empty and custom_keyword.strip():
        keyword_lower = custom_keyword.strip().lower()
        print(f'Filtering historic data with keyword: "{keyword_lower}"')

        # Apply filter
        filtered_historic = historic_df[historic_df.apply(
            lambda row: keyword_lower in str(row.get('title', '')).lower() or keyword_lower in str(row.get('intro', '')).lower(),
            axis=1,
        )]
        print(f'Historic articles after filtering: {len(filtered_historic)}')

        if not filtered_historic.empty:
            # Rename columns
            filtered_historic = filtered_historic.rename(columns={
                'date': 'published_at',
                'intro': 'description',
                'href': 'url',
            })
            filtered_historic['source'] = 'EconomicTimes Archive'
            filtered_historic['news_type'] = 'historic'

            # Concat
            news_df = pd.concat([news_df, filtered_historic], ignore_index=True, sort=False)
            print(f'Total merged articles: {len(news_df)}')

    # Analyze sentiment
    tickers = list_available_companies()[:10]
    sentiment_df = analyze_sentiment(news_df, text_column='title', tickers=tickers)
    print('Sentiment analysis completed')

    print('✅ Test passed!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()