import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

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
    ("ETERNAL.NS", "Eternal", "Consumer Tech"),
    ("JIOFIN.NS", "Jio Financial Services", "Finance"),
    ("CHOLAFIN.NS", "Cholamandalam Inv", "Finance"),
]


def fetch_stock_news(company_name, ticker_clean):
    try:
        query = urllib.parse.quote(f"{company_name} share price NSE")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:2]
        news_list = []
        bullish_words = ["surge", "jump", "gain", "profit", "buy", "upgrade", "rally", "order", "win", "high", "growth", "bullish", "beat"]
        bearish_words = ["fall", "drop", "loss", "sell", "downgrade", "slump", "cut", "low", "weak", "bearish", "miss", "probe", "penalty"]

        for item in items:
            title_el = item.find("title")
            pub_el = item.find("pubDate")
            title = title_el.text if title_el is not None and title_el.text else ""
            pub_date = pub_el.text if pub_el is not None and pub_el.text else ""
            if not title:
                continue
            title_lower = title.lower()
            sentiment = "NEUTRAL"
            if any(w in title_lower for w in bullish_words):
                sentiment = "POSITIVE"
            elif any(w in title_lower for w in bearish_words):
                sentiment = "NEGATIVE"
            parts = title.rsplit(" - ", 1)
            news_list.append({
                "ticker": ticker_clean,
                "headline": parts[0],
                "source": parts[1] if len(parts) > 1 else "Market News",
                "time": pub_date[:16],
                "sentiment": sentiment,
            })
        return news_list
    except Exception:
        return []


def run_scanner():
    results = []
    all_news = []
    processed = 0

    for symbol, name, sector in FO_STOCKS:
        try:
            if yf is None:
                continue
            df = yf.Ticker(symbol).history(period="1mo", interval="1d")
            if df is None or df.empty or len(df) < 8:
                continue

            df = df.dropna(subset=["Close", "High", "Low", "Volume"])
            if len(df) < 8:
                continue

            processed += 1
            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            curr_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            if prev_price <= 0:
                continue
            pct_change = ((curr_price - prev_price) / prev_price) * 100.0

            ema_span = min(20, len(close))
            ema20 = float(close.ewm(span=ema_span, adjust=False).mean().iloc[-1])

            vol_slice = volume.iloc[-11:-1] if len(volume) > 11 else volume.iloc[:-1]
            avg_vol = float(vol_slice.mean()) if len(vol_slice) > 0 else 1.0
            curr_vol = float(volume.iloc[-1])
            vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0

            daily_range = high - low
            curr_range = float(daily_range.iloc[-1])
            min_range_7 = float(daily_range.iloc[-7:].min())
            is_nr7 = curr_range <= min_range_7 * 1.05

            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=5).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=5).mean()
            rs = gain / loss.replace(0, float("nan"))
            last_rsi = rs.iloc[-1]
            rsi = float(100.0 - (100.0 / (1.0 + last_rsi))) if pd.notna(last_rsi) else 50.0

            score = 52
            reasons = []
            direction = "NEUTRAL"

            if curr_price > ema20 and pct_change >= 0.0:
                direction = "BULLISH"
                score += 16
                reasons.append(f"Trading above EMA (₹{ema20:.1f}) with positive bias")
            elif curr_price < ema20 and pct_change <= 0.0:
                direction = "BEARISH"
                score += 16
                reasons.append(f"Trading below EMA (₹{ema20:.1f}) showing weakness")
            elif is_nr7:
                direction = "BULLISH" if curr_price >= ema20 else "BEARISH"
                score += 18
                reasons.append("NR7 narrow range squeeze — breakout watch")

            if vol_ratio >= 1.2:
                score += 12
                reasons.append(f"Volume {vol_ratio:.1f}x vs recent average")
            if is_nr7:
                score += 10
                reasons.append("Tight range compression before potential expansion")
            if direction == "BULLISH" and 50 <= rsi <= 75:
                score += 8
                reasons.append(f"RSI at {rsi:.0f} supporting bullish momentum")
            elif direction == "BEARISH" and 25 <= rsi <= 50:
                score += 8
                reasons.append(f"RSI at {rsi:.0f} supporting bearish pressure")

            if direction == "NEUTRAL":
                continue

            clean_ticker = symbol.replace(".NS", "")
            day_high = float(high.iloc[-1])
            day_low = float(low.iloc[-1])

            step = 50 if curr_price > 2000 else (20 if curr_price > 1000 else (10 if curr_price > 400 else 5))
            atm_strike = int(round(curr_price / step) * step)

            if direction == "BULLISH":
                trigger = f"Above ₹{day_high:.1f}"
                option_idea = f"{atm_strike} CE"
                strategy = "Long Call / Bull Call Spread"
                stop_loss = f"₹{day_low:.1f}"
            else:
                trigger = f"Below ₹{day_low:.1f}"
                option_idea = f"{atm_strike} PE"
                strategy = "Long Put / Bear Put Spread"
                stop_loss = f"₹{day_high:.1f}"

            stock_news = fetch_stock_news(name, clean_ticker)
            all_news.extend(stock_news)

            results.append({
                "ticker": clean_ticker,
                "name": name,
                "sector": sector,
                "price": round(curr_price, 2),
                "changePct": round(pct_change, 2),
                "direction": direction,
                "confidence": min(score, 96),
                "setup": "NR7 Squeeze" if is_nr7 else ("Volume Surge" if vol_ratio >= 1.2 else "Trend Setup"),
                "trigger": trigger,
                "stopLoss": stop_loss,
                "optionIdea": option_idea,
                "strategy": strategy,
                "volRatio": round(vol_ratio, 2),
                "rsi": round(rsi, 1),
                "ema20": round(ema20, 2),
                "reasons": reasons,
                "news": stock_news,
            })
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    if not all_news:
        all_news = fetch_stock_news("NSE Nifty stocks India", "NIFTY")

    results.sort(key=lambda x: x["confidence"], reverse=True)

    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

    output = {
        "lastUpdated": now_ist,
        "totalScanned": processed if processed > 0 else len(FO_STOCKS),
        "stocks": results[:15],
        "newsFeed": all_news[:12],
    }

    with open("market_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Done. Processed {processed} stocks, saved {len(results[:15])} setups.")


if __name__ == "__main__":
    run_scanner()
    
