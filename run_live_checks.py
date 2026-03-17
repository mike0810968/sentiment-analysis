from src.data_preprocessing import list_available_companies
from src.live_data import live_news_data, live_stock_data
import os, json

# Stop interference: force fresh fetch
live_news_data.cache=None
live_news_data.last_update=0

tickers = list_available_companies()[:10]
print('Testing tickers:', tickers)

for t in tickers:
    company_name = None
    mapping_path = os.path.abspath('stock_market_project/src/company_mappings.json')
    if os.path.isfile(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as fh:
            mappings = json.load(fh)
        vals = mappings.get(t)
        if vals:
            company_name = vals[0]
    if not company_name:
        try:
            info = live_stock_data.get_current_price(t)
            company_name = info.get('company_name') if isinstance(info, dict) else None
        except Exception:
            company_name = None

    query = (company_name or t) + ' stock market NSE BSE financial news'
    print('\n=== Ticker:', t, 'Query:', query)
    df = live_news_data.get_live_news(query=query, limit=6)
    print('Articles fetched:', len(df))
    if df.empty:
        print('  (no live articles)')
        continue
    for i, row in df.head(3).iterrows():
        title = (row.get('title') or row.get('description') or '').strip()
        score = row.get('sentiment_score', None)
        label = row.get('sentiment_label', None)
        src = row.get('source', '')
        print(f"  - {title[:200]} | score={score} label={label} | source={src}")
