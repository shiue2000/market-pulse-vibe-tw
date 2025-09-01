import datetime
import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import openai
import plotly.graph_objs as go
import stripe
from dotenv import load_dotenv
import logging
import time
import twstock
import pandas as pd
import json, os

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
industry_mapping = {
    "半導體": "Semiconductors",
    "電子零組件": "Electronic Components",
    "電腦及週邊": "Computers and Peripherals",
    "金融保險": "Financial Services",
    "通信網路": "Communication Networks",
    "光電": "Optoelectronics",
    "汽車": "Automotive",
    "水泥": "Cement",
    "食品": "Food",
    "塑膠": "Plastics",
    "其他": "Others"
}
IMPORTANT_METRICS = [
    "pe", "pb", "roe", "roa", "gross_margin",
    "revenue_growth", "eps_growth", "debt_to_equity"
]
METRIC_NAMES_ZH_EN = {
    "pe": "本益比 (PE)",
    "pb": "股價淨值比 (PB)",
    "roe": "股東權益報酬率 (ROE)",
    "roa": "資產報酬率 (ROA)",
    "gross_margin": "毛利率 (Gross Margin)",
    "revenue_growth": "營收成長率 (YoY)",
    "eps_growth": "每股盈餘成長率 (EPS Growth YoY)",
    "debt_to_equity": "負債權益比 (Debt to Equity)"
}
QUOTE_FIELDS = {
    "current_price": ("即時股價", "Current Price"),
    "open": ("開盤價", "Open"),
    "high": ("最高價", "High"),
    "low": ("最低價", "Low"),
    "previous_close": ("前收盤價", "Previous Close"),
    "daily_change": ("漲跌幅(%)", "Change Percent"),
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

def get_quote(symbol):
    try:
        stock = twstock.Stock(symbol)
        quote_data = stock.fetch_from((datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m'), datetime.datetime.now().strftime('%Y-%m'))
        if not quote_data:
            return {}
        latest = quote_data[-1]
        quote = {
            'current_price': round(latest.close, 2),
            'open': round(latest.open, 2),
            'high': round(latest.high, 2),
            'low': round(latest.low, 2),
            'previous_close': round(quote_data[-2].close if len(quote_data) > 1 else latest.close, 2),
            'daily_change': round((latest.close - quote_data[-2].close) / quote_data[-2].close * 100, 2) if len(quote_data) > 1 else 'N/A',
            'volume': latest.capacity
        }
        return quote
    except Exception as e:
        logger.warning(f"[twstock Error] {symbol}: {e}")
        return {}

def get_metrics(symbol):
    try:
        stock = twstock.Stock(symbol)
        # twstock does not provide direct access to financial metrics like Finnhub
        # Simulate basic metrics using available data or external API if needed
        # For simplicity, return a subset of metrics (mocked or limited)
        metrics = {
            "pe": "N/A",  # Placeholder, requires external financial data source
            "pb": "N/A",
            "roe": "N/A",
            "roa": "N/A",
            "gross_margin": "N/A",
            "revenue_growth": "N/A",
            "eps_growth": "N/A",
            "debt_to_equity": "N/A"
        }
        return metrics
    except Exception as e:
        logger.warning(f"[twstock Metrics Error] {symbol}: {e}")
        return {}

def filter_metrics(metrics):
    filtered = {}
    for key in IMPORTANT_METRICS:
        v = metrics.get(key)
        if v != "N/A":
            try:
                v = float(v)
                if "growth" in key or "margin" in key or "roe" in key or "roa" in key:
                    filtered[key] = f"{v:.2f}%"
                else:
                    filtered[key] = round(v, 4)
            except:
                filtered[key] = str(v)
        else:
            filtered[key] = "N/A"
    return filtered

def get_recent_news(symbol):
    # twstock does not provide news; placeholder for external news API
    return []

def get_company_profile(symbol):
    try:
        stock = twstock.Stock(symbol)
        # twstock provides limited profile data; use twstock.codes for basic info
        profile = twstock.codes.get(symbol, {})
        return {
            "finnhubIndustry": profile.get("industry", "其他"),
            "name": profile.get("name", "未知")
        }
    except Exception as e:
        logger.warning(f"[twstock Profile Error] {symbol}: {e}")
        return {"finnhubIndustry": "其他"}

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_historical_data(symbol):
    try:
        stock = twstock.Stock(symbol)
        data = stock.fetch_from((datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m'), datetime.datetime.now().strftime('%Y-%m'))
        if not data:
            return pd.DataFrame(), {}
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'capacity']].rename(columns={'close': 'Close', 'capacity': 'Volume'})
        
        # Compute technical indicators
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        rsi = calculate_rsi(df['Close'])
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12.iloc[-1] - ema26.iloc[-1]
        support = df['Low'].tail(20).min()
        resistance = df['High'].tail(20).max()
        volume = df['Volume'].iloc[-1]
        technical = {
            'ma50': round(ma50, 2),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'volume': volume
        }
        return df, technical
    except Exception as e:
        logger.warning(f"[twstock Historical Error] {symbol}: {e}")
        return pd.DataFrame(), {}

def get_plot_html(df, symbol):
    if df.empty or 'Close' not in df.columns:
        return "<p class='text-danger'>📊 無法取得股價趨勢圖</p>"
    df_plot = df.tail(7)
    dates = df_plot.index.strftime('%Y-%m-%d').tolist()
    closes = df_plot['Close'].round(2).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, mode='lines+markers', name='Close Price'))
    fig.update_layout(
        title=f"{symbol} 最近7日收盤價 / 7-Day Closing Price Trend",
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
        try:
            session["request_count"] = request_count + 1
            quote = get_quote(symbol)
            metrics = filter_metrics(get_metrics(symbol))
            news = get_recent_news(symbol)
            profile = get_company_profile(symbol)
            industry_zh = profile.get("finnhubIndustry", "其他")
            industry_en = industry_mapping.get(industry_zh, "Others")
            df, technical = get_historical_data(symbol)
            quote['volume'] = technical.get('volume', 'N/A')
            plot_html = get_plot_html(df, symbol)
           
            technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items())
            prompt = f"請根據以下資訊產出中英文雙語股票分析: 股票代號: {symbol}, 目前價格: {quote.get('current_price','N/A')}, 產業分類: {industry_zh} ({industry_en}), 財務指標: {metrics}, 技術指標: {technical_str}. 請提供買入/賣出/持有建議."
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
                gpt_analysis = chat_response['choices'][0]['message']['content'].strip()
            if isinstance(gpt_analysis, str):
                gpt_analysis = {'summary': gpt_analysis + "\n\n---\n\n*以上分析僅供參考，投資有風險*"}
           
            result = {
                "symbol": symbol,
                "quote": quote,
                "industry_en": industry_en,
                "industry_zh": industry_zh,
                "metrics": metrics,
                "news": news,
                "gpt_analysis": gpt_analysis,
                "plot_html": plot_html,
                "technical": {k: v if v != 'N/A' else 'N/A' for k, v in technical.items()}
            }
        except Exception as e:
            result = {"error": f"資料讀取錯誤: {e}"}
            logger.error(f"Processing error for symbol {symbol}: {e}")
    return render_template("index.html",
                           result=result,
                           symbol_input=symbol,
                           QUOTE_FIELDS=QUOTE_FIELDS,
                           METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
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
