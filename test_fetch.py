from src.live_data import live_news_data, live_stock_data
import os, json

# Force fresh fetch
live_news_data.cache=None
live_news_data.last_update=0

ticker='RSSOFTWARE'
company_search_name=None
mapping_path=os.path.abspath('stock_market_project/src/company_mappings.json')
print('mapping_path',mapping_path)
if os.path.isfile(mapping_path):
    with open(mapping_path,'r',encoding='utf-8') as fh:
        mappings=json.load(fh)
    vals=mappings.get(ticker)
    print('mapping vals',vals)
    if vals:
        company_search_name=vals[0]
if not company_search_name:
    try:
        info=live_stock_data.get_current_price(ticker)
        print('live info', info.get('company_name'))
        company_search_name=info.get('company_name') if isinstance(info,dict) else None
    except Exception as e:
        print('live info err',e)
        company_search_name=None
query=(company_search_name or ticker)+' stock market NSE BSE financial news'
print('Using query:',query)
df=live_news_data.get_live_news(query=query,limit=10)
print('Articles:',len(df))
if not df.empty:
    for i,row in df.head(5).iterrows():
        print('-',row.get('title'))
