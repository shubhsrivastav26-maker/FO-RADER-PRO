import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Top 40 High-Liquidity NSE F&O Stocks
FO_STOCKS = [
    ("RELIANCE.NS", "Reliance Industries", "Energy"),
    ("HDFCBANK.NS", "HDFC Bank", "Banking"),
    ("ICICIBANK.NS", "ICICI Bank", "Banking"),
    ("INFY.NS", "Infosys", "IT"),
    ("TCS.NS", "Tata Consultancy Services", "IT"),
    ("SBIN.NS", "State Bank of India", "Banking"),
    ("BHARTIARTL.NS", "Bharti Airtel", "Telecom"),
    ("LT.NS", "Larsen & Toubro", "Infrastructure"),
    ("AXISBANK.NS", "Axis Bank", "Banking"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank", "Banking"),
    ("BAJFINANCE.NS", "Bajaj Finance", "Finance"),
    ("TATAMOTORS.NS", "Tata Motors", "Auto"),
    ("M&M.NS", "Mahindra & Mahindra", "Auto"),
    ("MARUTI.NS", "Maruti Suzuki", "Auto"),
    ("SUNPHARMA.NS", "Sun Pharma", "Pharma"),
    ("NTPC.NS", "NTPC", "Power"),
    ("POWERGRID.NS", "Power Grid Corp", "Power"),
    ("TATASTEEL.NS", "Tata Steel", "Metals"),
    ("HINDALCO.NS", "Hindalco Industries", "Metals"),
    ("JSWSTEEL.NS", "JSW Steel", "Metals"),
    ("BEL.NS", "Bharat Electronics", "Defence"),
    ("HAL.NS", "Hindustan Aeronautics", "Defence"),
    ("ASHOKLEY.NS", "Ashok Leyland", "Auto"),
    ("ADANIENT.NS", "Adani Enterprises", "Conglomerate"),
    ("ADANIPORTS.NS", "Adani Ports", "Infrastructure"),
    ("DLF.NS", "DLF", "Realty"),
    ("TRENT.NS", "Trent", "Retail"),
    ("TITAN.NS", "Titan Company", "Consumer"),
    ("WIPRO.NS", "Wipro", "IT"),
    ("HCLTECH.NS", "HCL Technologies", "IT"),
    ("TECHM.NS", "Tech Mahindra", "IT"),
    ("INDUSINDBK.NS", "IndusInd Bank", "Banking"),
    ("BANKBARODA.NS", "Bank of Baroda", "Banking"),
    ("PNB.NS", "Punjab National Bank", "Banking"),
    ("COALINDIA.NS", "Coal India", "Mining"),
    ("ONGC.NS", "ONGC", "Energy"),
    ("BPCL.NS", "BPCL", "Energy"),
    ("ZOMATO.NS", "Eternal (Zomato)", "Consumer Tech"),
    ("JIOFIN.NS", "Jio Financial Services", "Finance"),
    ("CHOLAFIN.NS", "Cholamandalam Inv", "Finance"),
]

def fetch_stock_news(company_name, ticker_clean):
    """Fetches real latest Google News RSS headlines for the stock."""
    try:
        query = urllib.parse.quote(f"{company_name} share price NSE")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')[:2]
        news_list = []
        bullish_words = ['surge', 'jump', 'gain', 'profit', 'buy', 'upgrade', 'rally', 'order', 'win', 'high', 'growth', 'bullish', 'beat']
        bearish_words = ['fall', 'drop', 'loss', 'sell', 'downgrade', 'slump', 'cut', 'low', 'weak', 'bearish', 'miss', 'probe', 'penalty']
        
        for item in items:
            title = item.find('title').text if item.find('title') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            title_lower = title.lower()
            sentiment = "NEUTRAL"
            if any(w in title_lower for w in bullish_words):
                sentiment = "POSITIVE"
            elif any(w in title_lower for w in bearish_words):
                sentiment = "NEGATIVE"
            news_list.append({
                "ticker": ticker_clean,
                "headline": title.split(" - ")[0],
                "source": title.split(" - ")[-1] if " - " in title else "Market News",
                "time": pub_date[:16],
                "sentiment": sentiment
            })
        return news_list
    except Exception:
        return []

def run_scanner():
    tickers = [s[0] for s in FO_STOCKS]
    print("Downloading real NSE market data...")
    data = yf.download(tickers, period="1mo", interval="1d", group_by="ticker", progress=False)
    
    results = []
    all_news = []

    for symbol, name, sector in FO_STOCKS:
        try:
            df = data[symbol].dropna()
            if len(df) < 10:
                continue

            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            curr_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            pct_change = ((curr_price - prev_price) / prev_price) * 100

            # Technical Indicators
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            avg_vol = float(volume.iloc[-11:-1].mean())
            curr_vol = float(volume.iloc[-1])
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

            # Range Compression (NR4 / NR7 Squeeze check)
            daily_range = high - low
            curr_range = float(daily_range.iloc[-1])
            min_range_7 = float(daily_range.iloc[-7:].min())
            is_nr7 = curr_range <= min_range_7 * 1.05

            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not pd.isna(rs.iloc[-1]) else 50.0

            # Scoring Logic for Bullish vs Bearish Pre-Breakout
            score = 50
            reasons = []
            direction = "NEUTRAL"

            if curr_price > ema20 and pct_change > 0.3:
                direction = "BULLISH"
                score += 18
                reasons.append(f"Trading above 20-EMA (₹{ema20:.1f}) with positive momentum")
            elif curr_price < ema20 and pct_change < -0.3:
                direction = "BEARISH"
                score += 18
                reasons.append(f"Trading below 20-EMA (₹{ema20:.1f}) showing distribution")
            elif is_nr7:
                direction = "BULLISH" if curr_price >= ema20 else "BEARISH"
                score += 20
                reasons.append("NR7 Narrow Range Coiling — High-probability volatility expansion due")

            if vol_ratio >= 1.3:
                score += 15
                reasons.append(f"Volume surge {vol_ratio:.1f}x above 10-day average")
            if is_nr7:
                score += 10
                reasons.append("Tight daily candle range squeeze before big move")
            if direction == "BULLISH" and 55 <= rsi <= 72:
                score += 10
                reasons.append(f"Strong bullish RSI ({rsi:.0f}) without extreme overbought risk")
            elif direction == "BEARISH" and 28 <= rsi <= 45:
                score
