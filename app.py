# -*- coding: utf-8 -*-
import os
import datetime
import time
import urllib.parse
from collections import namedtuple
import json
import requests
from flask import Flask, request, render_template, jsonify, redirect, session, flash, url_for
import stripe
import numpy as np
import logging
import plotly.express as px
import pandas as pd
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# ---------- Safe casting helpers ----------
def safe_int(val, default=0):
    """
    Convert val to int safely. Returns default on failure.
    Accepts numeric, numeric-strings, None.
    """
    try:
        # handle boolean explicitly
        if isinstance(val, bool):
            return int(val)
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        try:
            # try float->int fallback
            return int(float(val))
        except Exception:
            return default

def safe_float(val, default=0.0):
    """Convert val to float safely. Returns default on failure."""
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

# ---------- Basic checks & app setup ----------
# Ensure NumPy compatibility (clear error if >=2.0)
if not np.__version__.startswith("1."):
    raise RuntimeError(f"NumPy {np.__version__} is incompatible — please install numpy<2.0")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")
app.config["SESSION_PERMANENT"] = False

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Stripe config ----------
STRIPE_TEST_SECRET_KEY = os.getenv("STRIPE_TEST_SECRET_KEY")
STRIPE_TEST_PUBLISHABLE_KEY = os.getenv("STRIPE_TEST_PUBLISHABLE_KEY")
STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY")
STRIPE_LIVE_PUBLISHABLE_KEY = os.getenv("STRIPE_LIVE_PUBLISHABLE_KEY")
STRIPE_MODE = (os.getenv("STRIPE_MODE", "test") or "test").lower()
STRIPE_PRICE_IDS = {
    "Free": os.getenv("STRIPE_PRICE_TIER0"),
    "Basic": os.getenv("STRIPE_PRICE_TIER1"),
    "Pro": os.getenv("STRIPE_PRICE_TIER2"),
    "Premium": os.getenv("STRIPE_PRICE_TIER3"),
    "Enterprise": os.getenv("STRIPE_PRICE_TIER4"),
}
ENDPOINT_SECRET = os.getenv("STRIPE_ENDPOINT_SECRET")

if STRIPE_MODE == "live":
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY

if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
    raise RuntimeError(f"Stripe keys for mode '{STRIPE_MODE}' are not set in environment variables.")

stripe.api_key = STRIPE_SECRET_KEY

# ---------- OpenAI config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not set in environment (.env)")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- In-memory storage & constants ----------
user_data = {}

QUOTE_FIELDS = {
    "current_price": ("目前價格", "Current Price"),
    "daily_change": ("當日變化", "Daily Change (%)"),
    "volume": ("交易量", "Volume"),
    "open_price": ("開盤價", "Open Price"),
    "high_price": ("最高價", "High Price"),
    "low_price": ("最低價", "Low Price"),
    "prev_close": ("前日收盤價", "Previous Close"),
}

METRIC_NAMES_ZH_EN = {
    "pe": "本益比 (PE TTM) | PE Ratio (TTM)",
    "pb": "股價淨值比 (PB) | PB Ratio",
    "revenue_growth": "營收成長率 (YoY) | Revenue Growth (YoY)",
    "eps_growth": "每股盈餘成長率 (YoY) | EPS Growth (YoY)",
}

PRICING_TIERS = [
    {"name": "Free", "limit": 50, "price": 0},
    {"name": "Basic", "limit": 200, "price": 5},
    {"name": "Pro", "limit": 500, "price": 10},
    {"name": "Premium", "limit": 1000, "price": 20},
    {"name": "Enterprise", "limit": 3000, "price": 50},
]

# Finnhub
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    logger.warning("Finnhub API key not set — some features will degrade (names, metrics, news).")

STATIC_STOCK_NAMES = {
    "2330": "台積電 | Taiwan Semiconductor Manufacturing",
    "2317": "鴻海精密 | Hon Hai Precision",
    "2454": "聯發科 | MediaTek",
    "2382": "廣達電腦 | Quanta Computer",
}

industry_mapping = {
    "Semiconductors": "半導體",
    "Electronics": "電子",
    "Technology": "科技",
    "Healthcare": "醫療保健",
    "Financial Services": "金融服務",
    "Consumer Cyclical": "週期性消費品",
    "Industrials": "工業",
    "Consumer Defensive": "防禦性消費品",
    "Utilities": "公用事業",
    "Energy": "能源",
    "Communication Services": "通信服務",
    "Real Estate": "房地產",
    "Basic Materials": "基礎材料",
    "Unknown": "未知",
}

# twstock (optional)
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

try:
    from twstock import codes
except Exception:
    codes = {}
    logger.warning("twstock.codes not available — falling back to Finnhub/static names.")

TWSE_BASE_URL = "http://www.twse.com.tw/"
TPEX_BASE_URL = "http://www.tpex.org.tw/"
DATATUPLE = namedtuple(
    "Data",
    ["date", "capacity", "turnover", "open", "high", "low", "close", "change", "transaction"],
)

# ---------- Fetcher base and implementations ----------
class BaseFetcher:
    REPORT_URL = None

    def fetch(self, year: int, month: int, sid: str, retry: int = 5) -> dict:
        raise NotImplementedError

    def _convert_date(self, date_str: str) -> str:
        """Convert ROC date like '106/05/01' to '2017/05/01'"""
        if not date_str:
            raise ValueError("Empty date string")
        parts = date_str.split("/")
        if len(parts) < 3:
            raise ValueError(f"Unexpected date format: {date_str}")
        try:
            year = int(parts[0]) + 1911
            return "/".join([str(year)] + parts[1:])
        except Exception as e:
            logger.error(f"Date conversion error for '{date_str}': {e}")
            raise

    def _make_datatuple(self, data):
        raise NotImplementedError

class TWSEFetcher(BaseFetcher):
    REPORT_URL = urllib.parse.urljoin(TWSE_BASE_URL, "exchangeReport/STOCK_DAY")

    def fetch(self, year: int, month: int, sid: str, retry: int = 5) -> dict:
        params = {"date": f"{year}{month:02d}01", "stockNo": sid}
        for attempt in range(retry):
            try:
                r = requests.get(self.REPORT_URL, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if data.get("stat") != "OK":
                    logger.warning(f"TWSE response not OK for {sid}: stat={data.get('stat')}")
                    return {"stat": "", "data": []}
                pur = self.purify(data)
                logger.info(f"TWSE fetch success for {sid} {year}-{month}")
                return {"stat": "OK", "data": pur}
            except (requests.RequestException, JSONDecodeError) as e:
                logger.warning(f"TWSE fetch attempt {attempt+1} failed for {sid}: {e}")
                time.sleep(2)
        logger.error(f"TWSE fetch failed for {sid} after {retry} attempts")
        return {"stat": "", "data": []}

    def _make_datatuple(self, data):
        try:
            # Defensive copy
            row = list(data)
            row[0] = datetime.datetime.strptime(self._convert_date(row[0]), "%Y/%m/%d")
            row[1] = int(row[1].replace(",", "")) if row[1] and row[1] != "--" else None
            row[2] = int(row[2].replace(",", "")) if row[2] and row[2] != "--" else None
            row[3] = None if row[3] == "--" else float(row[3].replace(",", ""))
            row[4] = None if row[4] == "--" else float(row[4].replace(",", ""))
            row[5] = None if row[5] == "--" else float(row[5].replace(",", ""))
            row[6] = None if row[6] == "--" else float(row[6].replace(",", ""))
            # 'X0.00' indicates no change — normalize to 0.0
            change_raw = row[7].replace(",", "") if isinstance(row[7], str) else row[7]
            row[7] = float(0.0 if str(change_raw) == "X0.00" else change_raw)
            row[8] = int(row[8].replace(",", "")) if row[8] and row[8] != "--" else None
            return DATATUPLE(*row)
        except Exception as e:
            logger.error(f"Error making DATATUPLE (TWSE) for {data}: {e}")
            raise

    def purify(self, original_data):
        try:
            items = original_data.get("data", [])
            return [self._make_datatuple(d) for d in items]
        except Exception as e:
            logger.error(f"TWSE purify error: {e}")
            return []

class TPEXFetcher(BaseFetcher):
    REPORT_URL = urllib.parse.urljoin(TPEX_BASE_URL, "web/stock/aftertrading/daily_trading_info/st43_result.php")

    def fetch(self, year: int, month: int, sid: str, retry: int = 5) -> dict:
        params = {"d": f"{year-1911}/{month}", "stkno": sid}
        for attempt in range(retry):
            try:
                r = requests.get(self.REPORT_URL, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if data.get("aaData"):
                    pur = self.purify(data)
                    logger.info(f"TPEX fetch success for {sid} {year}-{month}")
                    return {"aaData": data.get("aaData", []), "data": pur}
                else:
                    logger.warning(f"TPEX returned no aaData for {sid}")
                    return {"aaData": [], "data": []}
            except (requests.RequestException, JSONDecodeError) as e:
                logger.warning(f"TPEX fetch attempt {attempt+1} failed for {sid}: {e}")
                time.sleep(2)
        logger.error(f"TPEX fetch failed for {sid} after {retry} attempts")
        return {"aaData": [], "data": []}

    def _convert_date(self, date_str: str) -> str:
        return super()._convert_date(date_str)

    def _make_datatuple(self, data):
        try:
            row = list(data)
            # Remove special markers and parse
            row[0] = datetime.datetime.strptime(self._convert_date(row[0].replace("＊", "")), "%Y/%m/%d")
            row[1] = int(row[1].replace(",", "")) * 1000 if row[1] and row[1] != "--" else None
            row[2] = int(row[2].replace(",", "")) * 1000 if row[2] and row[2] != "--" else None
            row[3] = None if row[3] == "--" else float(row[3].replace(",", ""))
            row[4] = None if row[4] == "--" else float(row[4].replace(",", ""))
            row[5] = None if row[5] == "--" else float(row[5].replace(",", ""))
            row[6] = None if row[6] == "--" else float(row[6].replace(",", ""))
            row[7] = float(row[7].replace(",", "")) if row[7] and row[7] != "--" else 0.0
            row[8] = int(row[8].replace(",", "")) if row[8] and row[8] != "--" else None
            return DATATUPLE(*row)
        except Exception as e:
            logger.error(f"Error making DATATUPLE (TPEX) for {data}: {e}")
            raise

    def purify(self, original_data):
        try:
            items = original_data.get("aaData", [])
            return [self._make_datatuple(d) for d in items]
        except Exception as e:
            logger.error(f"TPEX purify error: {e}")
            return []

# ---------- Stock wrapper ----------
class Stock:
    def __init__(self, sid: str, market: str = "上市"):
        self.sid = sid
        self.fetcher = TWSEFetcher() if market == "上市" else TPEXFetcher()
        self.raw_data = []
        self.data = []
        # Keep fetch lazy in case you want to control network calls from outside
        self.fetch_90()

    def _month_year_iter(self, start_month: int, start_year: int, end_month: int, end_year: int):
        ym_start = 12 * start_year + (start_month - 1)
        ym_end = 12 * end_year + end_month
        for ym in range(ym_start, ym_end):
            y, m = divmod(ym, 12)
            yield y, m + 1

    def fetch(self, year: int, month: int):
        try:
            resp = self.fetcher.fetch(year, month, self.sid)
            self.raw_data = [resp]
            self.data = resp.get("data", [])
            return self.data
        except Exception as e:
            logger.error(f"Stock.fetch error for {self.sid} {year}-{month}: {e}")
            self.data = []
            return []

    def fetch_from(self, year: int, month: int):
        self.raw_data = []
        self.data = []
        try:
            today = datetime.datetime.today()
            for y, m in self._month_year_iter(month, year, today.month, today.year):
                resp = self.fetcher.fetch(y, m, self.sid)
                self.raw_data.append(resp)
                self.data.extend(resp.get("data", []))
            return self.data
        except Exception as e:
            logger.error(f"Stock.fetch_from error for {self.sid}: {e}")
            self.data = []
            return []

    def fetch_90(self):
        try:
            today = datetime.datetime.today()
            before = today - datetime.timedelta(days=120)
            self.fetch_from(before.year, before.month)
            # keep last 90 records
            if len(self.data) > 90:
                self.data = self.data[-90:]
            return self.data
        except Exception as e:
            logger.error(f"Stock.fetch_90 error for {self.sid}: {e}")
            self.data = []
            return []

    # Properties
    @property
    def date(self):
        return [d.date for d in self.data]

    @property
    def capacity(self):
        return [d.capacity for d in self.data]

    @property
    def turnover(self):
        return [d.turnover for d in self.data]

    @property
    def price(self):
        return [d.close for d in self.data]

    @property
    def high(self):
        return [d.high for d in self.data]

    @property
    def low(self):
        return [d.low for d in self.data]

    @property
    def open(self):
        return [d.open for d in self.data]

    @property
    def close(self):
        return [d.close for d in self.data]

    @property
    def change(self):
        return [d.change for d in self.data]

    @property
    def transaction(self):
        return [d.transaction for d in self.data]

# ---------- Helpers: Finnhub and data lookups ----------
def get_finnhub_json(endpoint: str, params: dict) -> dict:
    """Robust wrapper for Finnhub API calls with retries."""
    if not FINNHUB_API_KEY:
        logger.error("Finnhub API key not set; returning empty response")
        return {}
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params = dict(params)
    params["token"] = FINNHUB_API_KEY
    for attempt in range(3):
        r = None
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            response = r.json()
            logger.info(f"Finnhub success: endpoint={endpoint} params={params}")
            return response
        except requests.HTTPError as e:
            status = getattr(r, "status_code", "N/A")
            if status == 403:
                logger.error(f"Finnhub 403 Forbidden for {endpoint} — key may be invalid or limited.")
            else:
                logger.warning(f"Finnhub HTTP error for {endpoint} attempt {attempt+1}: {e}")
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Finnhub request attempt {attempt+1} failed for {endpoint}: {e}")
            time.sleep(2)
    logger.error(f"Finnhub API failed after retries for {endpoint}")
    return {}

def get_stock_name(symbol: str) -> str:
    logger.info(f"Fetching stock name for: {symbol}")
    if symbol in STATIC_STOCK_NAMES:
        return STATIC_STOCK_NAMES[symbol]
    if codes and symbol in codes:
        try:
            return codes[symbol].name
        except Exception:
            logger.warning("twstock.codes lookup failed — falling back to Finnhub/static.")
    profile = get_finnhub_json("stock/profile2", {"symbol": f"{symbol}.TW"})
    return profile.get("name", "Unknown")

def get_quote(symbol: str) -> dict:
    try:
        stock = Stock(symbol, market="上市")
        market_label = "上市"
        if not stock.data:
            stock = Stock(symbol, market="上櫃")
            market_label = "上櫃"
            if not stock.data:
                logger.error(f"No data for {symbol} on TWSE or TPEX")
                return {"error": f"無法獲取股票 {symbol} 的數據 | No data available for {symbol}"}
        latest = stock.data[-1]
        prev = stock.data[-2] if len(stock.data) >= 2 else None
        current_price = latest.close
        prev_close = prev.close if prev else None
        daily_change = ((current_price - prev_close) / prev_close * 100) if (isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float))) else None
        return {
            "current_price": round(current_price, 2) if isinstance(current_price, (int, float)) else "N/A",
            "daily_change": f"{round(daily_change, 2)}%" if isinstance(daily_change, (int, float)) else "N/A",
            "volume": int(latest.capacity) if latest.capacity is not None else "N/A",
            "open_price": round(latest.open, 2) if isinstance(latest.open, (int, float)) else "N/A",
            "high_price": round(latest.high, 2) if isinstance(latest.high, (int, float)) else "N/A",
            "low_price": round(latest.low, 2) if isinstance(latest.low, (int, float)) else "N/A",
            "prev_close": round(prev_close, 2) if isinstance(prev_close, (int, float)) else "N/A",
            "market": market_label,
        }
    except Exception as e:
        logger.error(f"get_quote error for {symbol}: {e}")
        return {"error": f"無法獲取股票 {symbol} 的報價數據: {str(e)} | Failed to fetch quote for {symbol}: {str(e)}"}

def get_metrics(symbol: str) -> dict:
    try:
        metrics = {"pe": "N/A", "pb": "N/A", "revenue_growth": "N/A", "eps_growth": "N/A"}
        res = get_finnhub_json("stock/metric", {"symbol": f"{symbol}.TW"})
        metric = res.get("metric", {}) if isinstance(res, dict) else {}
        if metric:
            metrics["pe"] = round(metric.get("peTTM", 0), 2) if metric.get("peTTM") else "N/A"
            metrics["pb"] = round(metric.get("pb", 0), 2) if metric.get("pb") else "N/A"
        return metrics
    except Exception as e:
        logger.error(f"get_metrics error for {symbol}: {e}")
        return {"pe": "N/A", "pb": "N/A", "revenue_growth": "N/A", "eps_growth": "N/A"}

def filter_metrics(metrics: dict) -> dict:
    return {k: v for k, v in (metrics or {}).items() if v != "N/A"}

def get_company_profile(symbol: str) -> dict:
    try:
        return get_finnhub_json("stock/profile2", {"symbol": f"{symbol}.TW"}) or {}
    except Exception as e:
        logger.error(f"get_company_profile error for {symbol}: {e}")
        return {}

def get_historical_data(symbol: str):
    try:
        stock = Stock(symbol, market="上市")
        market_label = "上市"
        if not stock.data:
            stock = Stock(symbol, market="上櫃")
            market_label = "上櫃"
            if not stock.data:
                logger.error(f"No historical data for {symbol}")
                return pd.DataFrame(), {"ma50": "N/A", "support": "N/A", "resistance": "N/A", "volume": "N/A"}
        df = pd.DataFrame({
            "date": stock.date,
            "close": stock.close,
            "volume": stock.capacity,
        })
        prices = [p for p in stock.close if isinstance(p, (int, float))]
        if not prices:
            ma50 = support = resistance = "N/A"
        elif len(prices) >= 50:
            ma50 = float(np.mean(prices[-50:]))
            support = float(min(prices[-50:]))
            resistance = float(max(prices[-50:]))
        else:
            ma50 = float(np.mean(prices))
            support = float(min(prices))
            resistance = float(max(prices))
        technical = {
            "ma50": round(ma50, 2) if isinstance(ma50, (int, float)) else "N/A",
            "support": round(support, 2) if isinstance(support, (int, float)) else "N/A",
            "resistance": round(resistance, 2) if isinstance(resistance, (int, float)) else "N/A",
            "volume": int(stock.capacity[-1]) if stock.capacity and stock.capacity[-1] is not None else "N/A",
        }
        return df, technical
    except Exception as e:
        logger.error(f"get_historical_data error for {symbol}: {e}")
        return pd.DataFrame(), {"ma50": "N/A", "support": "N/A", "resistance": "N/A", "volume": "N/A"}

def get_plot_html(df: pd.DataFrame, symbol: str) -> str:
    if df.empty:
        logger.warning(f"No data to plot for {symbol}")
        return ""
    try:
        # Validate df
        if "date" not in df.columns or "close" not in df.columns:
            logger.error(f"DataFrame missing required columns for {symbol}")
            return ""
        if df["date"].isna().any() or df["close"].isna().any():
            logger.warning(f"DataFrame contains NaNs for {symbol}")
        fig = px.line(df, x="date", y="close", title=f"{symbol} Stock Price")
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (TWD)")
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"get_plot_html error for {symbol}: {e}")
        return ""

def get_recent_news(symbol: str):
    try:
        to_date = datetime.datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        news = get_finnhub_json("company-news", {"symbol": f"{symbol}.TW", "from": from_date, "to": to_date})
        if not isinstance(news, list):
            logger.warning(f"Company news not a list for {symbol}: {news}")
            return []
        # sort by datetime if present (unix timestamp expected)
        news_sorted = sorted(news, key=lambda x: x.get("datetime", 0) or 0, reverse=True)[:10]
        for n in news_sorted:
            try:
                n["datetime"] = datetime.datetime.utcfromtimestamp(n.get("datetime", 0)).strftime("%Y-%m-%d %H:%M")
            except Exception:
                n["datetime"] = "未知時間 | Unknown time"
        return news_sorted
    except Exception as e:
        logger.error(f"get_recent_news error for {symbol}: {e}")
        return []

# ---------- OpenAI wrapper ----------
def call_openai_for_analysis(prompt: str) -> dict:
    """
    Call OpenAI chat completion and parse JSON response.
    Uses a defensive approach with fallbacks on failure.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一位中英雙語金融分析助理，中英文內容完全對等。請以JSON格式回應，確保結構一致且無錯誤。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=0.6,
        )
        # Extract content safely
        content = ""
        try:
            content = response.choices[0].message.content
        except Exception:
            content = getattr(response.choices[0], "text", "")
        try:
            parsed = json.loads(content)
            return parsed
        except Exception as e:
            logger.warning(f"OpenAI returned non-JSON or parse failed: {e}; returning fallback text.")
            return {
                "recommendation": "hold",
                "rationale": "OpenAI response parse failed. Fallback used.",
                "risk": "中等風險，請保持謹慎。| Moderate risk, use caution.",
                "summary": "OpenAI 回傳不可解析內容，採用保守建議。| OpenAI returned unparseable content; conservative recommendation applied.",
            }
    except (AuthenticationError, RateLimitError, APIError) as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "recommendation": "hold",
            "rationale": f"OpenAI API 錯誤: {str(e)}。採用後備邏輯。",
            "risk": "中等風險，監控市場動態。",
            "summary": "OpenAI API 發生錯誤，採用後備建議。",
        }
    except Exception as e:
        logger.error(f"Unexpected OpenAI error: {e}")
        return {
            "recommendation": "hold",
            "rationale": f"分析失敗: {str(e)}，採用後備邏輯。",
            "risk": "中等風險，監控市場動態。",
            "summary": "分析失敗，採用後備建議。",
        }

# ---------- Main aggregator ----------
def get_stock_data(symbol: str) -> dict:
    # Input validation
    if not isinstance(symbol, str) or not symbol.isdigit() or len(symbol) != 4:
        return {"error": "股票代號必須為4位數字 | Stock ID must be a 4-digit number"}

    logger.info(f"Processing stock data for {symbol}")

    try:
        stock_name = get_stock_name(symbol)
        quote = get_quote(symbol)
        if "error" in quote:
            return quote

        metrics = filter_metrics(get_metrics(symbol))
        news = get_recent_news(symbol)
        profile = get_company_profile(symbol)
        industry_en = profile.get("finnhubIndustry", "Unknown")
        industry_zh = industry_mapping.get(industry_en, "未知")
        df, technical = get_historical_data(symbol)
        plot_html = get_plot_html(df, symbol)

        # Prepare prompt
        technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items())
        metrics_items = metrics if metrics else {}
        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics_items.items()) if metrics_items else "無可用財務指標 | No financial metrics available"

        prompt = f"""
請根據以下資訊產出中英文雙語股票分析:
股票代號: {symbol},
股票名稱: {stock_name},
目前價格: {quote.get('current_price', 'N/A')},
產業分類: {industry_zh} ({industry_en}),
財務指標: {metrics_str},
技術指標: {technical_str}.
請提供：
1. 投資建議 (買入/賣出/持有 | Buy/Sell/Hold)
2. 理由 (Rationale)
3. 風險評估 (Risk Assessment)
4. 總結 (Summary)
回答需以JSON格式回應，包含中英文內容完全對等：
{{ 
    "recommendation": "buy" or "sell" or "hold",
    "rationale": "中文 rationale\\nEnglish rationale",
    "risk": "中文 risk\\nEnglish risk",
    "summary": "中文 summary\\nEnglish summary"
}}
"""
        gpt_analysis = call_openai_for_analysis(prompt)
        technical_clean = {k: str(v) if v != "N/A" else "N/A" for k, v in technical.items()}

        return {
            "symbol": symbol,
            "stock_name": stock_name,
            "quote": quote,
            "industry_en": industry_en,
            "industry_zh": industry_zh,
            "metrics": metrics,
            "news": news,
            "gpt_analysis": gpt_analysis,
            "plot_html": plot_html,
            "technical": technical_clean,
            "market": quote.get("market", "上市"),
        }
    except Exception as e:
        logger.error(f"get_stock_data unexpected error for {symbol}: {e}")
        return {"error": f"無法獲取股票 {symbol} 的數據: {str(e)} | Failed to fetch data for {symbol}: {str(e)}"}

# ---------- Flask routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    symbol_input = ""
    result = {}
    user_id = request.remote_addr or "unknown"

    # Use safe_int for session values (may be strings)
    current_tier_index = safe_int(session.get("paid_tier", 0), 0)
    current_tier_index = min(max(current_tier_index, 0), len(PRICING_TIERS) - 1)
    current_tier = PRICING_TIERS[current_tier_index]
    current_tier_name = current_tier["name"]

    # Get request_count from session first, fallback to in-memory user_data
    request_count = safe_int(session.get("request_count", user_data.get(f"user:{user_id}:request_count", 0)), 0)
    current_limit = safe_int(current_tier.get("limit", 0), 0)

    if request.method == "POST":
        symbol = (request.form.get("symbol") or "").strip()
        symbol_input = symbol
        if not symbol:
            result = {"error": "請輸入股票代號 | Please enter a stock symbol"}
        elif request_count >= current_limit:
            result = {"error": f"已達到 {current_tier_name} 方案的請求限制 ({current_limit}) | Request limit reached for {current_tier_name} tier ({current_limit})"}
        else:
            result = get_stock_data(symbol)
            if "error" not in result:
                # increment both session and in-memory user_data safely
                request_count = request_count + 1
                session["request_count"] = request_count
                user_data[f"user:{user_id}:request_count"] = request_count

    return render_template(
        "index.html",
        symbol_input=symbol_input,
        result=result,
        current_tier_name=current_tier_name,
        request_count=request_count,
        current_limit=current_limit,
        QUOTE_FIELDS=QUOTE_FIELDS,
        METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
        tiers=PRICING_TIERS,
        stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
        stripe_mode=STRIPE_MODE,
    )

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    # tier can be passed as tier name or index — try name first
    tier_name = request.form.get("tier")
    tier = next((t for t in PRICING_TIERS if t["name"] == tier_name), None)
    # fallback: treat as index
    if not tier:
        tier_index = safe_int(tier_name, None)
        if tier_index is not None and 0 <= tier_index < len(PRICING_TIERS):
            tier = PRICING_TIERS[tier_index]
    if not tier:
        logger.error("Invalid tier requested: %s", tier_name)
        return jsonify({"error": "無效的方案 | Invalid tier"}), 400

    if tier["name"] == "Free":
        session["subscribed"] = False
        session["paid_tier"] = 0
        session["request_count"] = 0
        user_data[f"user:{request.remote_addr}:tier"] = "Free"
        user_data[f"user:{request.remote_addr}:request_count"] = 0
        flash("✅ 已切換到免費方案 | Switched to Free tier.", "success")
        return jsonify({"url": url_for("index", _external=True)})

    price_id = STRIPE_PRICE_IDS.get(tier["name"])
    if not price_id:
        logger.error("No Price ID configured for %s", tier["name"])
        flash(f"⚠️ {tier['name']} 方案目前不可用 | Subscription for {tier['name']} is currently unavailable.", "warning")
        return jsonify({"error": f"{tier['name']} 方案目前不可用 | Subscription for {tier['name']} is currently unavailable"}), 400

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=url_for("payment_success", tier_name=tier["name"], _external=True),
            cancel_url=url_for("index", _external=True),
            metadata={"tier": tier["name"], "user_id": request.remote_addr},
        )
        return jsonify({"url": checkout_session.url})
    except Exception as e:
        logger.exception("Stripe checkout creation failed")
        return jsonify({"error": f"無法創建結帳會話: {str(e)} | Failed to create checkout session: {str(e)}"}), 500

@app.route("/payment-success/<tier_name>")
def payment_success(tier_name):
    tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t["name"] == tier_name), None)
    if tier_index is not None and tier_name != "Free":
        session["subscribed"] = True
        session["paid_tier"] = int(tier_index)
        session["request_count"] = 0
        user_data[f"user:{request.remote_addr}:tier"] = tier_name
        user_data[f"user:{request.remote_addr}:request_count"] = 0
        flash(f"✅ 成功訂閱 {tier_name} 方案 | Subscription successful for {tier_name} plan.", "success")
        logger.info("Subscription updated for %s", tier_name)
    return redirect(url_for("index"))

@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature")
    try:
        # If you haven't configured ENDPOINT_SECRET, skip verification (not recommended for production)
        if ENDPOINT_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, ENDPOINT_SECRET)
        else:
            event = json.loads(payload)
        if event.get("type") == "checkout.session.completed":
            stripe_session = event["data"]["object"]
            metadata = stripe_session.get("metadata", {}) if isinstance(stripe_session, dict) else {}
            user_id = metadata.get("user_id")
            tier = metadata.get("tier")
            tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t["name"] == tier), None)
            if tier_index is not None and user_id:
                user_data[f"user:{user_id}:tier"] = tier
                user_data[f"user:{user_id}:request_count"] = 0
                logger.info("Webhook: Updated %s to %s tier", user_id, tier)
    except Exception as e:
        logger.exception("Webhook processing failed")
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "success"})

@app.route("/reset", methods=["POST"])
def reset():
    password = request.form.get("password")
    if password == os.getenv("RESET_PASSWORD"):
        user_id = request.remote_addr
        session["request_count"] = 0
        session["subscribed"] = False
        session["paid_tier"] = 0
        user_data[f"user:{user_id}:request_count"] = 0
        user_data[f"user:{user_id}:tier"] = "Free"
        flash("✅ 請求次數已重置 | Request count reset.", "success")
        logger.info("Reset request count for user %s", user_id)
    else:
        flash("❌ 密碼錯誤 | Incorrect password.", "danger")
        logger.warning("Failed reset attempt")
    return redirect(url_for("index"))

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=safe_int(os.getenv("PORT", 8080), 8080))
