# -*- coding: utf-8 -*-
import datetime
import urllib.parse
from collections import namedtuple
import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import openai
import plotly.graph_objs as go
import stripe
from dotenv import load_dotenv
import logging
import pandas as pd
import json, os

# ------------------ TWStock Code ------------------
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

# Proxy support (if needed)
def get_proxies():
    return None  # Placeholder; implement proxy logic if required

# TWStock codes (simplified placeholder; assumes codes dictionary exists)
codes = {
    "2330": {"market": "上市", "name": "台積電"},  # Example: TSMC
    "2317": {"market": "上市", "name": "鴻海"},    # Example: Hon Hai
    # Add more Taiwan stock codes as needed
}

TWSE_BASE_URL = "http://www.twse.com.tw/"
TPEX_BASE_URL = "http://www.tpex.org.tw/"

DATATUPLE = namedtuple(
    "Data",
    [
        "date",
        "capacity",
        "turnover",
        "open",
        "high",
        "low",
        "close",
        "change",
        "transaction",
    ],
)

class BaseFetcher(object):
    def fetch(self, year, month, sid, retry):
        pass

    def _convert_date(self, date):
        """Convert '106/05/01' to '2017/05/01'"""
        return "/".join([str(int(date.split("/")[0]) + 1911)] + date.split("/")[1:])

    def _make_datatuple(self, data):
        pass

    def purify(self, original_data):
        pass

class TWSEFetcher(BaseFetcher):
    REPORT_URL = urllib.parse.urljoin(TWSE_BASE_URL, "exchangeReport/STOCK_DAY")

    def fetch(self, year: int, month: int, sid: str, retry: int = 5):
        params = {"date": "%d%02d01" % (year, month), "stockNo": sid}
        for retry_i in range(retry):
            r = requests.get(self.REPORT_URL, params=params, proxies=get_proxies())
            try:
                data = r.json()
            except JSONDecodeError:
                continue
            else:
                break
        else:
            data = {"stat": "", "data": []}
        if data["stat"] == "OK":
            data["data"] = self.purify(data)
        else:
            data["data"] = []
        return data

    def _make_datatuple(self, data):
        data[0] = datetime.datetime.strptime(self._convert_date(data[0]), "%Y/%m/%d")
        data[1] = int(data[1].replace(",", ""))
        data[2] = int(data[2].replace(",", ""))
        data[3] = None if data[3] == "--" else float(data[3].replace(",", ""))
        data[4] = None if data[4] == "--" else float(data[4].replace(",", ""))
        data[5] = None if data[5] == "--" else float(data[5].replace(",", ""))
        data[6] = None if data[6] == "--" else float(data[6].replace(",", ""))
        data[7] = float(
            0.0 if data[7].replace(",", "") == "X0.00" else data[7].replace(",", "")
        )
        data[8] = int(data[8].replace(",", ""))
        return DATATUPLE(*data)

    def purify(self, original_data):
        return [self._make_datatuple(d) for d in original_data["data"]]

class TPEXFetcher(BaseFetcher):
    REPORT_URL = urllib.parse.urljoin(
        TPEX_BASE_URL, "web/stock/aftertrading/daily_trading_info/st43_result.php"
    )

    def fetch(self, year: int, month: int, sid: str, retry: int = 5):
        params = {"d": "%d/%d" % (year - 1911, month), "stkno": sid}
        for retry_i in range(retry):
            r = requests.get(self.REPORT_URL, params=params, proxies=get_proxies())
            try:
                data = r.json()
            except JSONDecodeError:
                continue
            else:
                break
        else:
            data = {"aaData": []}
        data["data"] = []
        if data["aaData"]:
            data["data"] = self.purify(data)
        return data

    def _convert_date(self, date):
        return "/".join([str(int(date.split("/")[0]) + 1911)] + date.split("/")[1:])

    def _make_datatuple(self, data):
        data[0] = datetime.datetime.strptime(
            self._convert_date(data[0].replace("＊", "")), "%Y/%m/%d"
        )
        data[1] = int(data[1].replace(",", "")) * 1000
        data[2] = int(data[2].replace(",", "")) * 1000
        data[3] = None if data[3] == "--" else float(data[3].replace(",", ""))
        data[4] = None if data[4] == "--" else float(data[4].replace(",", ""))
        data[5] = None if data[5] == "--" else float(data[5].replace(",", ""))
        data[6] = None if data[6] == "--" else float(data[6].replace(",", ""))
        data[7] = float(data[7].replace(",", ""))
        data[8] = int(data[8].replace(",", ""))
        return DATATUPLE(*data)

    def purify(self, original_data):
        return [self._make_datatuple(d) for d in original_data["aaData"]]

class Stock:
    def __init__(self, sid: str, initial_fetch: bool = True):
        self.sid = sid
        self.fetcher = TWSEFetcher() if codes[sid].market == "上市" else TPEXFetcher()
        self.raw_data = []
        self.data = []
        if initial_fetch:
            self.fetch_31()

    def _month_year_iter(self, start_month, start_year, end_month, end_year):
        ym_start = 12 * start_year + start_month - 1
        ym_end = 12 * end_year + end_month
        for ym in range(ym_start, ym_end):
            y, m = divmod(ym, 12)
            yield y, m + 1

    def fetch(self, year: int, month: int):
        self.raw_data = [self.fetcher.fetch(year, month, self.sid)]
        self.data = self.raw_data[0]["data"]
        return self.data

    def fetch_from(self, year: int, month: int):
        self.raw_data = []
        self.data = []
        today = datetime.datetime.today()
        for year, month in self._month_year_iter(month, year, today.month, today.year):
            self.raw_data.append(self.fetcher.fetch(year, month, self.sid))
            self.data.extend(self.raw_data[-1]["data"])
        return self.data

    def fetch_31(self):
        today = datetime.datetime.today()
        before = today - datetime.timedelta(days=60)
        self.fetch_from(before.year, before.month)
        self.data = self.data[-31:]
        return self.data

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

# ------------------ Load environment ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")

# Stripe keys
STRIPE_TEST_SECRET_KEY = os.getenv("STRIPE_TEST_SECRET_KEY")
STRIPE_TEST_PUBLISHABLE_KEY = os.getenv("STRIPE_TEST_PUBLISHABLE_KEY")
STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY")
STRIPE_LIVE_PUBLISHABLE_KEY = os.getenv("STRIPE_LIVE_PUBLISHABLE_KEY")
STRIPE_MODE = os.getenv("STRIPE_MODE", "test").lower()

# Stripe Price IDs
STRIPE_PRICE_IDS = {
    "Free": os.getenv("STRIPE_PRICE_TIER0"),
    "Tier 1": os.getenv("STRIPE_PRICE_TIER1"),
    "Tier 2": os.getenv("STRIPE_PRICE_TIER2"),
    "Tier 3": os.getenv("STRIPE_PRICE_TIER3"),
    "Tier 4": os.getenv("STRIPE_PRICE_TIER4"),
}

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not set in .env")

# Set Stripe keys
if STRIPE_MODE == "live":
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY

if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
    raise RuntimeError(f"❌ Stripe keys for mode '{STRIPE_MODE}' not set in .env")

stripe.api_key = STRIPE_SECRET_KEY

# ------------------ Logger setup ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Initialize Flask & OpenAI ------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY
openai.api_key = OPENAI_API_KEY

# ------------------ Stock app config ------------------
QUOTE_FIELDS = {
    "current_price": ("即時股價", "Current Price"),
    "open": ("開盤價", "Open"),
    "high": ("最高價", "High"),
    "low": ("最低價", "Low"),
    "previous_close": ("前收盤價", "Previous Close"),
    "daily_change": ("漲跌幅", "Change"),
    "volume": ("交易量", "Volume")
}

# ------------------ Stripe pricing tiers ------------------
PRICING_TIERS = [
    {"name": "Free", "limit": 50, "price": 0},
    {"name": "Tier 1", "limit": 100, "price": 9.99},
    {"name": "Tier 2", "limit": 200, "price": 19.99},
    {"name": "Tier 3", "limit": 400, "price": 29.99},
    {"name": "Tier 4", "limit": 800, "price": 39.99},
]

# ------------------ Helper functions ------------------
def validate_price_id(price_id, tier_name):
    return bool(price_id)

def get_stock_data(sid):
    try:
        stock = Stock(sid)
        if not stock.data:
            return None, None
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame({
            'Date': stock.date,
            'Open': stock.open,
            'High': stock.high,
            'Low': stock.low,
            'Close': stock.close,
            'Volume': stock.capacity,
            'Change': stock.change
        }).set_index('Date')
        
        # Calculate technical indicators
        ma50 = df['Close'].rolling(window=50, min_periods=1).mean().iloc[-1]
        support = df['Low'].tail(20).min()
        resistance = df['High'].tail(20).max()
        volume = df['Volume'].iloc[-1]
        previous_close = df['Close'].shift(1).iloc[-1] if len(df) > 1 else None
        
        quote = {
            'current_price': stock.close[-1] if stock.close else 'N/A',
            'open': stock.open[-1] if stock.open else 'N/A',
            'high': stock.high[-1] if stock.high else 'N/A',
            'low': stock.low[-1] if stock.low else 'N/A',
            'previous_close': previous_close if previous_close else 'N/A',
            'daily_change': stock.change[-1] if stock.change else 'N/A',
            'volume': volume if volume else 'N/A'
        }
        
        technical = {
            'ma50': round(ma50, 2) if pd.notnull(ma50) else 'N/A',
            'support': round(support, 2) if pd.notnull(support) else 'N/A',
            'resistance': round(resistance, 2) if pd.notnull(resistance) else 'N/A',
            'volume': volume if volume else 'N/A'
        }
        
        return df, quote, technical
    except Exception as e:
        logger.error(f"Error fetching stock data for {sid}: {e}")
        return None, None, None

def get_plot_html(df, sid):
    if df is None or df.empty or 'Close' not in df.columns:
        return "<p class='text-danger'>📊 無法取得股價趨勢圖</p>"
    df_plot = df.tail(7)
    dates = df_plot.index.strftime('%Y-%m-%d').tolist()
    closes = df_plot['Close'].round(2).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, mode='lines+markers', name='Close Price'))
    fig.update_layout(
        title=f"{sid} 最近7日收盤價 / 7-Day Closing Price Trend",
        xaxis_title="日期 / Date",
        yaxis_title="收盤價 (TWD)",
        template="plotly_white",
        height=400
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height="400px", default_width="100%")

# ------------------ Flask routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    symbol = ""
    current_tier_index = session.get("paid_tier", 0)
    current_tier = PRICING_TIERS[current_tier_index]
    request_count = session.get("request_count", 0)
    current_limit = current_tier["limit"]
    current_tier_name = current_tier["name"]

    if request.method == "POST":
        if request_count >= current_limit:
            result["error"] = f"已達 {current_tier_name} 等級請求上限，請升級方案"
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        symbol = request.form.get("symbol", "").strip()
        if not symbol:
            result["error"] = "請輸入股票代號 / Please enter a stock symbol"
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        if symbol not in codes:
            result["error"] = f"無效的股票代號 {symbol} / Invalid stock symbol {symbol}"
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        try:
            session["request_count"] = request_count + 1
            df, quote, technical = get_stock_data(symbol)
            if df is None:
                result["error"] = f"無法取得 {symbol} 的股票資料 / Unable to fetch data for {symbol}"
                return render_template("index.html", result=result, symbol_input=symbol,
                                       tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                       stripe_mode=STRIPE_MODE, request_count=request_count,
                                       current_tier_name=current_tier_name, current_limit=current_limit)

            plot_html = get_plot_html(df, symbol)
            technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items() if v != 'N/A')
            company_name = codes[symbol].get("name", "未知")
            prompt = f"請根據以下資訊產出中英文雙語股票分析: 股票代號: {symbol}, 公司名稱: {company_name}, 目前價格: {quote.get('current_price','N/A')}, 技術指標: {technical_str}. 請提供買入/賣出/持有建議."
            chat_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一位中英雙語金融分析助理，中英文內容完全對等。請以JSON格式回應: {'recommendation': 'buy' or 'sell' or 'hold', 'rationale': '中文 rationale\\nEnglish rationale', 'risk': '中文 risk\\nEnglish risk', 'summary': '中文 summary\\nEnglish summary'}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=999,
                temperature=0.6,
                response_format={"type": "json_object"}
            )
            try:
                gpt_analysis = json.loads(chat_response['choices'][0]['message']['content'])
            except:
                gpt_analysis = {'summary': chat_response['choices'][0]['message']['content'].strip() + "\n\n---\n\n*以上分析僅供參考，投資有風險*"}

            result = {
                "symbol": symbol,
                "company_name": company_name,
                "quote": quote,
                "technical": {k: v if v != 'N/A' else 'N/A' for k, v in technical.items()},
                "gpt_analysis": gpt_analysis,
                "plot_html": plot_html
            }
        except Exception as e:
            result = {"error": f"資料讀取錯誤: {e}"}
            logger.error(f"Processing error for symbol {symbol}: {e}")

    return render_template("index.html",
                           result=result,
                           symbol_input=symbol,
                           QUOTE_FIELDS=QUOTE_FIELDS,
                           tiers=PRICING_TIERS,
                           stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                           stripe_mode=STRIPE_MODE,
                           request_count=request_count,
                           current_tier_name=current_tier_name,
                           current_limit=current_limit)

# ------------------ Stripe & Subscription Routes ------------------
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    tier_name = request.form.get("tier")
    tier = next((t for t in PRICING_TIERS if t["name"] == tier_name), None)
    if not tier:
        logger.error(f"Invalid tier requested: {tier_name}")
        return jsonify({"error": "Invalid tier"}), 400

    if tier["name"] == "Free":
        session["subscribed"] = False
        session["paid_tier"] = 0
        session["request_count"] = 0
        flash("✅ Switched to Free tier.", "success")
        return jsonify({"url": url_for("index", _external=True)})

    price_id = STRIPE_PRICE_IDS.get(tier_name)
    if not price_id or not validate_price_id(price_id, tier_name):
        logger.error(f"No valid Price ID configured for {tier_name}")
        flash(f"⚠️ Subscription for {tier_name} is currently unavailable.", "warning")
        return jsonify({"error": f"Subscription for {tier_name} is currently unavailable"}), 400

    try:
        logger.info(f"Creating Stripe checkout session for {tier_name} with Price ID: {price_id}")
        session_stripe = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=url_for('payment_success', tier_name=tier_name, _external=True),
            cancel_url=url_for('index', _external=True)
        )
        return jsonify({"url": session_stripe.url})
    except Exception as e:
        logger.error(f"Unexpected Stripe error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/payment-success/<tier_name>")
def payment_success(tier_name):
    tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t["name"] == tier_name), None)
    if tier_index is not None and tier_name != "Free":
        session["subscribed"] = True
        session["paid_tier"] = tier_index
        session["request_count"] = 0
        flash(f"✅ Subscription successful for {tier_name} plan.", "success")
        logger.info(f"Subscription successful for {tier_name} (tier index: {tier_index})")
    return redirect(url_for("index"))

@app.route("/reset", methods=["POST"])
def reset():
    password = request.form.get("password")
    if password == "888888":
        session["request_count"] = 0
        session["subscribed"] = False
        session["paid_tier"] = 0
        flash("✅ Counts reset.", "success")
        logger.info("Session counts reset successfully")
    else:
        flash("❌ Incorrect password.", "danger")
        logger.warning("Failed reset attempt with incorrect password")
    return redirect(url_for("index"))

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
