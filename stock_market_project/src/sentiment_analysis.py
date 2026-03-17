import os
from typing import List, Optional

import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional fuzzy matching (RapidFuzz). If unavailable, the code will fall back
# to exact / simple matching.
try:
    from rapidfuzz import fuzz as _rfuzz
except Exception:
    _rfuzz = None


DATA_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, "..")
ECONOMIC_TIMES_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "..", "economictimes_data.csv")
)


def load_news_data(path: str = None) -> pd.DataFrame:
    """Load Economic Times news dataset."""
    if path is None:
        path = ECONOMIC_TIMES_CSV

    if not os.path.isfile(path):
        raise FileNotFoundError(f"News CSV not found at {path}")

    # Try to load with UTF-8; fall back to common encodings if there are decoding issues.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure we have a date column
    if "date" in df.columns:
        # Handle the specific date format: "Sep 29, 2016, 10:38 AM IST"
        # First, remove the "IST" part and parse
        df["date"] = df["date"].str.replace(r'\s+IST$', '', regex=True)
        df["date"] = pd.to_datetime(df["date"], format='%b %d, %Y, %I:%M %p', errors="coerce")
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def find_related_companies(text: str, tickers: List[str]) -> List[str]:
    """Find a list of matching tickers in the provided text."""
    if not isinstance(text, str):
        return []

    text_up = text.upper()

    # Create a mapping of common company name variations to tickers
    company_mappings = {
        # Nifty 50 companies - common variations
        'HDFC': ['HDFC', 'HDFCBANK', 'HDFC BANK'],
        'ICICI': ['ICICI', 'ICICIBANK', 'ICICI BANK'],
        'KOTAK': ['KOTAK', 'KOTAKBANK', 'KOTAK MAHINDRA', 'KOTAK MAHINDRA BANK'],
        'AXIS': ['AXIS', 'AXISBANK', 'AXIS BANK'],
        'SBI': ['SBI', 'STATE BANK', 'STATE BANK OF INDIA'],
        'BAJAJ': ['BAJAJ', 'BAJAJ FINANCE', 'BAJAJFINSV', 'BAJAJ AUTO', 'BAJAJ-AUTO'],
        'RELIANCE': ['RELIANCE', 'RELIANCE INDUSTRIES'],
        'TCS': ['TCS', 'TATA CONSULTANCY', 'TATA CONSULTANCY SERVICES'],
        'INFY': ['INFY', 'INFOSYS'],
        'ITC': ['ITC', 'ITC LIMITED'],
        'LT': ['LT', 'LARSEN', 'LARSEN & TOUBRO'],
        'WIPRO': ['WIPRO'],
        'MARUTI': ['MARUTI', 'MARUTI SUZUKI'],
        'TATAMOTORS': ['TATA MOTORS', 'TATAMOTORS'],
        'BHARTI': ['BHARTI', 'BHARTI AIRTEL', 'BHARTIARTL'],
        'SUNPHARMA': ['SUN PHARMA', 'SUNPHARMA'],
        'CIPLA': ['CIPLA'],
        'DRREDDY': ['DR REDDY', 'DRREDDY', 'DR REDDYS'],
        'HEROMOTOCO': ['HERO MOTOCORP', 'HEROMOTOCO', 'HERO'],
        'GRASIM': ['GRASIM', 'GRASIM INDUSTRIES'],
        'ULTRACEMCO': ['ULTRATECH CEMENT', 'ULTRACEMCO', 'ULTRATECH'],
        'NTPC': ['NTPC'],
        'POWERGRID': ['POWER GRID', 'POWERGRID'],
        'ONGC': ['ONGC', 'OIL AND NATURAL GAS'],
        'COALINDIA': ['COAL INDIA', 'COALINDIA'],
        'GAIL': ['GAIL', 'GAIL INDIA'],
        'BPCL': ['BPCL', 'BHARAT PETROLEUM'],
        'IOC': ['IOC', 'INDIAN OIL'],
        'JSWSTEEL': ['JSW STEEL', 'JSWSTEEL'],
        'TATASTEEL': ['TATA STEEL', 'TATASTEEL'],
        'VEDL': ['VEDANTA', 'VEDL'],
        'HINDALCO': ['HINDALCO', 'HINDALCO INDUSTRIES'],
        'ADANIPORTS': ['ADANI PORTS', 'ADANIPORTS'],
        'SHREECEM': ['SHREE CEMENT', 'SHREECEM'],
        'BAJFINANCE': ['BAJAJ FINANCE', 'BAJFINANCE'],
        'TECHM': ['TECH MAHINDRA', 'TECHM'],
        'DIVISLAB': ['DIVI S LAB', 'DIVISLAB'],
        'HINDUNILVR': ['HINDUSTAN UNILEVER', 'HINDUNILVR'],
        'NESTLEIND': ['NESTLE', 'NESTLEIND', 'NESTLE INDIA'],
        'BRITANNIA': ['BRITANNIA'],
        'TITAN': ['TITAN'],
        'APOLLOHOSP': ['APOLLO HOSPITALS', 'APOLLOHOSP'],
        'ADANIENT': ['ADANI ENTERPRISES', 'ADANIENT'],
        'ADANIGREEN': ['ADANI GREEN', 'ADANIGREEN'],
        'ADANIPOWER': ['ADANI POWER', 'ADANIPOWER'],
        'ATGL': ['ADANI TOTAL GAS', 'ATGL'],
        'M&M': ['MAHINDRA', 'MAHINDRA & MAHINDRA', 'MM'],
        'INDUSINDBK': ['INDUSIND BANK', 'INDUSINDBK'],
        'HCLTECH': ['HCL', 'HCLTECH', 'HCL TECHNOLOGIES'],
        'EICHERMOT': ['EICHER MOTORS', 'EICHERMOT'],
        'SBIN': ['SBI', 'STATE BANK OF INDIA', 'SBIN'],
        'BAJAJFINSV': ['BAJAJ FINSERV', 'BAJAJFINSV'],
        'BAJAJ-AUTO': ['BAJAJ AUTO', 'BAJAJ-AUTO'],
        'UPL': ['UPL', 'UPL LIMITED'],
        'PIDILITIND': ['PIDILITE', 'PIDILITIND'],
        'BERGEPAINT': ['BERGER PAINTS', 'BERGEPAINT'],
        'DABUR': ['DABUR'],
        'HAVELLS': ['HAVELLS'],
        'GODREJCP': ['GODREJ CONSUMER', 'GODREJCP'],
        'MCDOWELL-N': ['MCDOWELL', 'MCDOWELL-N'],
        'IGL': ['INDRAPRASTHA GAS', 'IGL'],
        'PAGEIND': ['PAGE INDUSTRIES', 'PAGEIND'],
        'MARICO': ['MARICO'],
        'COLPAL': ['COLGATE', 'COLPAL'],
        'NMDC': ['NMDC'],
        'ZYDUSLIFE': ['ZYDUS LIFESCIENCES', 'ZYDUSLIFE'],
        'BIOCON': ['BIOCON'],
        'PEL': ['PIRAMAL ENTERPRISES', 'PEL'],
        'AMBUJACEM': ['AMBUJA CEMENT', 'AMBUJACEM'],
        'ACC': ['ACC', 'ACC LIMITED'],
        'DMART': ['DMART', 'AVENUE SUPERMARTS'],
        'JUBLFOOD': ['JUBILANT FOODWORKS', 'JUBLFOOD'],
        'INDIGO': ['INDIGO', 'INDIGO PAINTS'],
        'LUPIN': ['LUPIN'],
        'AUROPHARMA': ['AUROBINDO PHARMA', 'AUROPHARMA'],
        'ALKEM': ['ALKEM LABORATORIES', 'ALKEM'],
        'TORNTPHARM': ['TORRENT PHARMA', 'TORNTPHARM'],
        'GLENMARK': ['GLENMARK PHARMA', 'GLENMARK'],
        'IPCALAB': ['IPCA LABORATORIES', 'IPCALAB'],
        'LAURUSLABS': ['LAURUS LABS', 'LAURUSLABS'],
        'CHOLAFIN': ['CHOLAMANDALAM', 'CHOLAFIN'],
        'BANDHANBNK': ['BANDHAN BANK', 'BANDHANBNK', 'BANDHAN'],
        'IDFCFIRSTB': ['IDFC FIRST BANK', 'IDFCFIRSTB'],
        'PNB': ['PUNJAB NATIONAL BANK', 'PNB'],
        'BANKBARODA': ['BANK OF BARODA', 'BANKBARODA'],
        'CANBK': ['CANARA BANK', 'CANBK'],
        'UNIONBANK': ['UNION BANK', 'UNIONBANK'],
        'IOB': ['INDIAN OVERSEAS BANK', 'IOB'],
        'FEDERALBNK': ['FEDERAL BANK', 'FEDERALBNK'],
        'RBLBANK': ['RBL BANK', 'RBLBANK'],
        'IDBI': ['IDBI BANK', 'IDBI'],
        'CUB': ['CITY UNION BANK', 'CUB'],
    }

    related = []

    # First, match exact tickers
    for ticker in tickers:
        if not isinstance(ticker, str):
            continue
        t_up = ticker.upper()
        t_base = t_up.replace('.NS', '')
        # Match either the full ticker or the base ticker without exchange suffix
        if (t_up in text_up or t_base in text_up) and ticker not in related:
            related.append(ticker)

    # Then try company name variations
    for ticker, variations in company_mappings.items():
        if ticker in tickers:
            for variation in variations:
                if variation.upper() in text_up and ticker not in related:
                    related.append(ticker)
                    break

    # Fallback: check if any known ticker as a whole word (to avoid substring mismatches)
    if not related:
        import re
        words = set(re.findall(r"\b[A-Z0-9&-]{2,}\b", text_up))
        for ticker in tickers:
            try:
                tup = ticker.upper()
                tup_base = tup.replace('.NS', '')
            except Exception:
                continue
            if (tup in words or tup_base in words) and ticker not in related:
                related.append(ticker)

    # If still not found and RapidFuzz is available, try fuzzy matching against
    # a curated company mappings file (company_mappings.json) to catch name
    # variations and partial matches.
    if not related and _rfuzz is not None:
        try:
            mappings_path = os.path.join(os.path.dirname(__file__), "company_mappings.json")
            if os.path.isfile(mappings_path):
                with open(mappings_path, "r", encoding="utf-8") as fh:
                    mappings = json.load(fh)

                # Threshold can be tuned; 85 is a conservative default
                THRESHOLD = 85
                for ticker, variations in mappings.items():
                    if ticker not in tickers:
                        continue
                    for var in variations:
                        try:
                            score = _rfuzz.token_set_ratio(text_up, var.upper())
                        except Exception:
                            score = 0
                        if score >= THRESHOLD and ticker not in related:
                            related.append(ticker)
                            break
        except Exception:
            # If mapping load or fuzzy matching fails, quietly continue
            pass

    return related


def find_related_company(text: str, tickers: List[str]) -> Optional[str]:
    """Return the first matching ticker found in the text."""
    companies = find_related_companies(text, tickers)
    return companies[0] if companies else None


def analyze_sentiment(
    df: pd.DataFrame,
    text_column: str = "title",
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add VADER sentiment scores and labels to news DataFrame.

    This function includes a small finance-specific adjustment layer to better
    capture sentiment in earnings/market headlines.
    """
    analyzer = SentimentIntensityAnalyzer()

    # Finance-specific sentiment adjustment rules (phrase -> adjustment)
    adjustment_rules = {
        # Positive indicators
        "beats expectations": 0.15,
        "beats estimates": 0.15,
        "beats forecast": 0.12,
        "record profit": 0.18,
        "strong quarter": 0.12,
        "raises guidance": 0.15,
        "upgrades": 0.12,
        "buyback": 0.08,
        "dividend": 0.06,
        "positive outlook": 0.14,
        # Negative indicators
        "misses expectations": -0.18,
        "misses estimates": -0.18,
        "misses forecast": -0.15,
        "downgrade": -0.14,
        "cuts guidance": -0.16,
        "profit warning": -0.18,
        "loss": -0.12,
        "weak quarter": -0.12,
        "layoffs": -0.14,
        "delays": -0.10,
        "lawsuit": -0.10,
        "regulatory" : -0.08,
    }

    def classify(score: float) -> str:
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    def adjust_score(text: str, base_score: float) -> float:
        """Adjust VADER score using finance-specific phrase rules."""
        text_lower = text.lower()
        adjusted = base_score
        for phrase, delta in adjustment_rules.items():
            if phrase in text_lower:
                adjusted += delta
        # Keep within [-1, 1]
        return max(-1.0, min(1.0, adjusted))

    df = df.copy()
    # Build a combined text field (title + description + content) for better context
    def combined_text(row):
        parts = []
        parts.append(str(row.get(text_column, '') or ''))
        parts.append(str(row.get('description', '') or ''))
        parts.append(str(row.get('content', '') or ''))
        return ' '.join([p for p in parts if p])

    df['_combined_text'] = df.apply(combined_text, axis=1)
    df["_vader"] = df["_combined_text"].astype(str).map(analyzer.polarity_scores)
    df["sentiment_score"] = df["_vader"].map(lambda x: x.get("compound", 0.0))

    # Apply finance-specific adjustments
    df["sentiment_score"] = df.apply(
        lambda row: adjust_score(f"{row.get(text_column, '')}", row["sentiment_score"]),
        axis=1,
    )

    df["sentiment_label"] = df["sentiment_score"].map(classify)

    if tickers is not None:
        # Use combined text (title+description+content) for company matching
        df["related_companies"] = df["_combined_text"].astype(str).map(
            lambda t: find_related_companies(t, tickers)
        )
        # Keep backward-compatible single-company column for existing UI
        df["related_company"] = df["related_companies"].map(lambda x: x[0] if isinstance(x, list) and x else None)
    # Clean up helper columns
    drop_cols = ["_vader", "_combined_text"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df
