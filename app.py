# -*- coding: utf-8 -*-
import os
import datetime
import time
import urllib.parse
from collections import namedtuple
import json
import requests
from flask import Flask, request, render_template, jsonify, redirect, session, flash
import stripe
import numpy as np
import logging
import plotly.express as px
import pandas as pd
from openai import OpenAI

# Ensure NumPy version is <2.0 for compatibility
assert np.__version__.startswith("1."), "NumPy version must be <2.0"

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'supersecretkey')

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stripe setup
STRIPE_TEST_SECRET_KEY = os.getenv('STRIPE_TEST_SECRET_KEY')
STRIPE_TEST_PUBLISHABLE_KEY = os.getenv('STRIPE_TEST_PUBLISHABLE_KEY')
STRIPE_LIVE_SECRET_KEY = os.getenv('STRIPE_LIVE_SECRET_KEY')
STRIPE_LIVE_PUBLISHABLE_KEY = os.getenv('STRIPE_LIVE_PUBLISHABLE_KEY')
STRIPE_MODE = os.getenv('STRIPE_MODE', 'test').lower()
STRIPE_PRICE_IDS = {
    'Free': os.getenv('STRIPE_PRICE_TIER0'),
    'Basic': os.getenv('STRIPE_PRICE_TIER1'),
    'Pro': os.getenv('STRIPE_PRICE_TIER2'),
    'Premium': os.getenv('STRIPE_PRICE_TIER3'),
    'Enterprise': os.getenv('STRIPE_PRICE_TIER4'),
}
ENDPOINT_SECRET = os.getenv('STRIPE_ENDPOINT_SECRET')
if STRIPE_MODE == 'live':
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY
if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
    raise RuntimeError(f"❌ Stripe keys for mode '{STRIPE_MODE}' not set in .env")
stripe.api_key = STRIPE_SECRET_KEY

# OpenAI setup
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OpenAI API key not set in .env")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# In-memory storage for user data
user_data = {}

# Constants
QUOTE_FIELDS = {
    'current_price': ('目前價格', 'Current Price'),
    'daily_change': ('當日變化', 'Daily Change (%)'),
    'volume': ('交易量', 'Volume'),
    'open_price': ('開盤價', 'Open Price'),
    'high_price': ('最高價', 'High Price'),
    'low_price': ('最低價', 'Low Price'),
    'prev_close': ('前日收盤價', 'Previous Close')
}
METRIC_NAMES_ZH_EN = {
    'pe': '本益比 (PE TTM) | PE Ratio (TTM)',
    'pb': '股價淨值比 (PB) | PB Ratio',
    'revenue_growth': '營收成長率 (YoY) | Revenue Growth (YoY)',
    'eps_growth': '每股盈餘成長率 (YoY) | EPS Growth (YoY)'
}
PRICING_TIERS = [
    {'name': 'Free', 'limit': 50, 'price': 0},
    {'name': 'Basic', 'limit': 200, 'price': 5},
    {'name': 'Pro', 'limit': 500, 'price': 10},
    {'name': 'Premium', 'limit': 1000, 'price': 20},
    {'name': 'Enterprise', 'limit': 3000, 'price': 50}
]

# Finnhub setup
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
if not FINNHUB_API_KEY:
    logger.warning("Finnhub API key not set, stock name lookup may fail")

# Static fallback for common stocks
STATIC_STOCK_NAMES = {
    '2330': '台積電 | Taiwan Semiconductor Manufacturing',
    '2317': '鴻海精密 | Hon Hai Precision',
    '2454': '聯發科 | MediaTek'
}

# Industry mapping (basic, can be expanded)
industry_mapping = {
    'Semiconductors': '半導體',
    'Electronics': '電子',
    'Technology': '科技',
    'Unknown': '未知'
}

# TWStock code integrated
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

try:
    from twstock import codes
except ImportError:
    codes = {}
    logger.warning("twstock.codes not available, relying on Finnhub or static fallback")

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
            r = requests.get(self.REPORT_URL, params=params, timeout=10)
            try:
                data = r.json()
            except JSONDecodeError:
                logger.warning(f"TWSE fetch failed for {sid}, attempt {retry_i + 1}")
                time.sleep(2)
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
        data[7] = float(0.0 if data[7].replace(",", "") == "X0.00" else data[7].replace(",", ""))
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
            r = requests.get(self.REPORT_URL, params=params, timeout=10)
            try:
                data = r.json()
            except JSONDecodeError:
                logger.warning(f"TPEX fetch failed for {sid}, attempt {retry_i + 1}")
                time.sleep(2)
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
    def __init__(self, sid: str, market: str = "上市"):
        self.sid = sid
        self.fetcher = TWSEFetcher() if market == "上市" else TPEXFetcher()
        self.raw_data = []
        self.data = []
        self.fetch_90()
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
    def fetch_90(self):
        today = datetime.datetime.today()
        before = today - datetime.timedelta(days=120)
        self.fetch_from(before.year, before.month)
        self.data = self.data[-90:]
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

def get_stock_name(symbol):
    logger.info(f"Attempting to fetch stock name for symbol: {symbol}")
    if symbol in STATIC_STOCK_NAMES:
        logger.info(f"Using static stock name for {symbol}: {STATIC_STOCK_NAMES[symbol]}")
        return STATIC_STOCK_NAMES[symbol]
    if codes and symbol in codes:
        logger.info(f"Found stock name in twstock for {symbol}: {codes[symbol].name}")
        return codes[symbol].name
    profile = get_finnhub_json("stock/profile2", {"symbol": f"{symbol}.TW"})
    stock_name = profile.get("name", "Unknown")
    logger.info(f"Finnhub response for {symbol}: {profile}, stock name: {stock_name}")
    return stock_name

def get_finnhub_json(endpoint, params):
    if not FINNHUB_API_KEY:
        logger.error("Finnhub API key not set, returning empty response")
        return {}
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params["token"] = FINNHUB_API_KEY
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            response = r.json()
            logger.info(f"Finnhub API success for {endpoint}, params: {params}, response: {response}")
            return response
        except Exception as e:
            logger.warning(f"Finnhub API failed for {endpoint}, attempt {attempt + 1}: {e}")
            time.sleep(2)
    logger.error(f"Finnhub API failed after 3 attempts for {endpoint}")
    return {}

def get_quote(symbol):
    stock = Stock(symbol, market="上市")
    market = "上市"
    if not stock.data:
        stock = Stock(symbol, market="上櫃")
        market = "上櫃"
        if not stock.data:
            logger.error(f"No data available for {symbol} on TWSE or TPEX")
            return {'error': f'無法獲取股票 {symbol} 的數據 | No data available for {symbol}'}
    latest_data = stock.data[-1]
    prev_data = stock.data[-2] if len(stock.data) >= 2 else None
    current_price = latest_data.close
    prev_close = prev_data.close if prev_data else None
    daily_change = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else None
    return {
        'current_price': round(current_price, 2) if current_price else 'N/A',
        'daily_change': round(daily_change, 2) if daily_change else 'N/A',
        'volume': int(latest_data.capacity) if latest_data.capacity else 'N/A',
        'open_price': round(latest_data.open, 2) if latest_data.open else 'N/A',
        'high_price': round(latest_data.high, 2) if latest_data.high else 'N/A',
        'low_price': round(latest_data.low, 2) if latest_data.low else 'N/A',
        'prev_close': round(prev_close, 2) if prev_close else 'N/A'
    }

def get_metrics(symbol):
    metrics = {
        'pe': 'N/A',
        'pb': 'N/A',
        'revenue_growth': 'N/A',
        'eps_growth': 'N/A'
    }
    finnhub_metrics = get_finnhub_json("stock/metric", {"symbol": f"{symbol}.TW"})
    if finnhub_metrics.get('metric'):
        metrics['pe'] = round(finnhub_metrics['metric'].get('peTTM', 'N/A'), 2) if finnhub_metrics['metric'].get('peTTM') else 'N/A'
        metrics['pb'] = round(finnhub_metrics['metric'].get('pb', 'N/A'), 2) if finnhub_metrics['metric'].get('pb') else 'N/A'
    return metrics

def filter_metrics(metrics):
    return {k: v for k, v in metrics.items() if v != 'N/A'}

def get_company_profile(symbol):
    profile = get_finnhub_json("stock/profile2", {"symbol": f"{symbol}.TW"})
    return profile

def get_historical_data(symbol):
    stock = Stock(symbol, market="上市")
    market = "上市"
    if not stock.data:
        stock = Stock(symbol, market="上櫃")
        market = "上櫃"
        if not stock.data:
            logger.error(f"No historical data for {symbol}")
            return pd.DataFrame(), {'ma50': 'N/A', 'support': 'N/A', 'resistance': 'N/A', 'volume': 'N/A'}
    df = pd.DataFrame({
        'date': stock.date,
        'close': stock.close,
        'volume': stock.capacity
    })
    prices = [p for p in stock.close if p is not None]
    if len(prices) >= 50:
        ma50 = np.mean(prices[-50:])
        support = min(prices[-50:])
        resistance = max(prices[-50:])
    elif prices:
        ma50 = np.mean(prices)
        support = min(prices)
        resistance = max(prices)
    else:
        ma50 = support = resistance = 'N/A'
    technical = {
        'ma50': round(ma50, 2) if isinstance(ma50, (int, float)) else 'N/A',
        'support': round(support, 2) if isinstance(support, (int, float)) else 'N/A',
        'resistance': round(resistance, 2) if isinstance(support, (int, float)) else 'N/A',
        'volume': int(stock.capacity[-1]) if stock.capacity else 'N/A'
    }
    return df, technical

def get_plot_html(df, symbol):
    if df.empty:
        return ""
    fig = px.line(df, x='date', y='close', title=f"{symbol} Stock Price")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (TWD)")
    return fig.to_html(full_html=False)

def get_recent_news(symbol):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    past = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    news = get_finnhub_json("company-news", {"symbol": f"{symbol}.TW", "from": past, "to": today})
    if not isinstance(news, list):
        logger.warning(f"No news data for {symbol}, received: {news}")
        return []
    news = sorted(news, key=lambda x: x.get("datetime", 0), reverse=True)[:10]
    for n in news:
        try:
            n["datetime"] = datetime.datetime.utcfromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M")
        except:
            n["datetime"] = "未知時間 | Unknown time"
    return news

def get_stock_data(symbol):
    if not symbol.isdigit() or len(symbol) != 4:
        logger.error(f"Invalid stock symbol: {symbol}")
        return {'error': '股票代號必須為4位數字 | Stock ID must be a 4-digit number'}
    
    try:
        stock_name = get_stock_name(symbol)
        quote = get_quote(symbol)
        if 'error' in quote:
            return quote
        metrics = filter_metrics(get_metrics(symbol))
        news = get_recent_news(symbol)
        profile = get_company_profile(symbol)
        industry_en = profile.get("finnhubIndustry", "Unknown")
        industry_zh = industry_mapping.get(industry_en, "未知")
        df, technical = get_historical_data(symbol)
        plot_html = get_plot_html(df, symbol)
        
        technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items())
        prompt = f"""
        請根據以下資訊產出中英文雙語股票分析: 
        股票代號: {symbol}, 
        目前價格: {quote.get('current_price', 'N/A')}, 
        產業分類: {industry_zh} ({industry_en}), 
        財務指標: {metrics}, 
        技術指標: {technical_str}.
        請提供：
        1. 投資建議 (買入/賣出/持有 | Buy/Sell/Hold)
        2. 理由 (Rationale)
        3. 風險評估 (Risk Assessment)
        4. 總結 (Summary)
        回答需以JSON格式回應，包含中英文內容完全對等：
        {'recommendation': 'buy' or 'sell' or 'hold', 
         'rationale': '中文 rationale\\nEnglish rationale', 
         'risk': '中文 risk\\nEnglish risk', 
         'summary': '中文 summary\\nEnglish summary'}
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一位中英雙語金融分析助理，中英文內容完全對等。請以JSON格式回應，確保結構一致且無錯誤。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=999,
            temperature=0.6,
            response_format={"type": "json_object"}
        )
        try:
            gpt_analysis = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            gpt_analysis = {
                'recommendation': 'hold',
                'rationale': '分析失敗，採用後備邏輯。\nAnalysis failed, using fallback logic.',
                'risk': '中等風險，需密切關注市場動態。\nModerate risk, monitor market dynamics closely.',
                'summary': '由於分析失敗，建議謹慎並持續關注市場。\nDue to analysis failure, exercise caution and monitor market trends.'
            }
        
        return {
            'symbol': symbol,
            'stock_name': stock_name,
            'quote': quote,
            'industry_en': industry_en,
            'industry_zh': industry_zh,
            'metrics': metrics,
            'news': news,
            'gpt_analysis': gpt_analysis,
            'plot_html': plot_html,
            'technical': {k: v if v != 'N/A' else 'N/A' for k, v in technical.items()},
            'market': '上市' if not df.empty else '上櫃'
        }
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return {'error': f'無法獲取股票 {symbol} 的數據: {str(e)} | Failed to fetch data for {symbol}: {str(e)}'}

@app.route('/', methods=['GET', 'POST'])
def index():
    symbol_input = ''
    result = {}
    user_id = request.remote_addr
    current_tier_index = session.get('paid_tier', 0)
    current_tier = PRICING_TIERS[current_tier_index]
    current_tier_name = current_tier['name']
    request_count = session.get('request_count', user_data.get(f'user:{user_id}:request_count', 0))
    current_limit = current_tier['limit']
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip()
        symbol_input = symbol
        if not symbol:
            result = {'error': '請輸入股票代號 | Please enter a stock symbol'}
        elif request_count >= current_limit:
            result = {'error': f'已達到 {current_tier_name} 方案的請求限制 ({current_limit}) | Request limit reached for {current_tier_name} tier ({current_limit})'}
        else:
            result = get_stock_data(symbol)
            if 'error' not in result:
                session['request_count'] = request_count + 1
                user_data[f'user:{user_id}:request_count'] = session['request_count']
    return render_template(
        'index.html',
        symbol_input=symbol_input,
        result=result,
        current_tier_name=current_tier_name,
        request_count=request_count,
        current_limit=current_limit,
        QUOTE_FIELDS=QUOTE_FIELDS,
        METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
        tiers=PRICING_TIERS,
        stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
        stripe_mode=STRIPE_MODE
    )

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    tier_name = request.form.get('tier')
    tier = next((t for t in PRICING_TIERS if t['name'] == tier_name), None)
    if not tier:
        logger.error(f"Invalid tier requested: {tier_name}")
        return jsonify({'error': '無效的方案 | Invalid tier'}), 400
    if tier['name'] == 'Free':
        session['subscribed'] = False
        session['paid_tier'] = 0
        session['request_count'] = 0
        user_data[f'user:{request.remote_addr}:tier'] = 'Free'
        user_data[f'user:{request.remote_addr}:request_count'] = 0
        flash('✅ 已切換到免費方案 | Switched to Free tier.', 'success')
        return jsonify({'url': url_for('index', _external=True)})
    price_id = STRIPE_PRICE_IDS.get(tier_name)
    if not price_id:
        logger.error(f"No valid Price ID configured for {tier_name}")
        flash(f'⚠️ {tier_name} 方案目前不可用 | Subscription for {tier_name} is currently unavailable.', 'warning')
        return jsonify({'error': f'{tier_name} 方案目前不可用 | Subscription for {tier_name} is currently unavailable'}), 400
    try:
        logger.info(f"Creating Stripe checkout session for {tier_name} with Price ID: {price_id}")
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=url_for('payment_success', tier_name=tier_name, _external=True),
            cancel_url=url_for('index', _external=True),
            metadata={'tier': tier_name, 'user_id': request.remote_addr}
        )
        return jsonify({'url': checkout_session.url})
    except Exception as e:
        logger.error(f"Unexpected Stripe error: {e}")
        return jsonify({'error': f'無法創建結帳會話: {str(e)} | Failed to create checkout session: {str(e)}'}), 500

@app.route('/payment-success/<tier_name>')
def payment_success(tier_name):
    tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t['name'] == tier_name), None)
    if tier_index is not None and tier_name != 'Free':
        session['subscribed'] = True
        session['paid_tier'] = tier_index
        session['request_count'] = 0
        user_data[f'user:{request.remote_addr}:tier'] = tier_name
        user_data[f'user:{request.remote_addr}:request_count'] = 0
        flash(f'✅ 成功訂閱 {tier_name} 方案 | Subscription successful for {tier_name} plan.', 'success')
        logger.info(f"Subscription successful for {tier_name} (tier index: {tier_index})")
    return redirect(url_for('index'))

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, ENDPOINT_SECRET)
        if event['type'] == 'checkout.session.completed':
            stripe_session = event['data']['object']
            user_id = stripe_session['metadata']['user_id']
            tier = stripe_session['metadata']['tier']
            tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t['name'] == tier), None)
            if tier_index is not None:
                session['subscribed'] = True
                session['paid_tier'] = tier_index
                session['request_count'] = 0
                user_data[f'user:{user_id}:tier'] = tier
                user_data[f'user:{user_id}:request_count'] = 0
                logger.info(f"Webhook: Updated {user_id} to {tier} tier")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 400
    return jsonify({'status': 'success'})

@app.route('/reset', methods=['POST'])
def reset():
    password = request.form.get('password')
    if password == os.getenv('RESET_PASSWORD'):
        user_id = request.remote_addr
        session['request_count'] = 0
        session['subscribed'] = False
        session['paid_tier'] = 0
        user_data[f'user:{user_id}:request_count'] = 0
        user_data[f'user:{user_id}:tier'] = 'Free'
        flash('✅ 請求次數已重置 | Request count reset.', 'success')
        logger.info(f"Reset request count for user {user_id}")
    else:
        flash('❌ 密碼錯誤 | Incorrect password.', 'danger')
        logger.warning('Failed reset attempt with incorrect password')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
