# -*- coding: utf-8 -*-
import datetime
import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import openai
import plotly.graph_objs as go
import stripe
from dotenv import load_dotenv
import logging
import time
import yfinance as yf
import pandas as pd
import json, os

# Import Taiwan stock modules
from proxy import get_proxies
from analytics import Analytics, BestFourPoint
from stock import Stock, TWSEFetcher, TPEXFetcher, DATATUPLE
from realtime import get, get_raw

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------ Initialize Flask & OpenAI ------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY
openai.api_key = OPENAI_API_KEY

# ------------------ Stock app config ------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
industry_mapping = {
    "Technology": "科技業",
    "Financial Services": "金融服務業",
    "Healthcare": "醫療保健業",
    "Consumer Cyclical": "非必需消費品業",
    "Communication Services": "通訊服務業",
    "Energy": "能源業",
    "Industrials": "工業類股",
    "Utilities": "公用事業",
    "Real Estate": "房地產業",
    "Materials": "原物料業",
    "Consumer Defensive": "必需消費品業",
    "Unknown": "未知"
}
IMPORTANT_METRICS = [
    "peTTM", "pb", "roeTTM", "roaTTM", "grossMarginTTM",
    "revenueGrowthTTMYoy", "epsGrowthTTMYoy", "debtToEquityAnnual"
]
METRIC_NAMES_ZH_EN = {
    "pe_ratio": "本益比 (PE TTM)",
    "pb_ratio": "股價淨值比 (PB)",
    "roe_ttm": "股東權益報酬率 (ROE TTM)",
    "roa_ttm": "資產報酬率 (ROA TTM)",
    "gross_margin_ttm": "毛利率 (Gross Margin TTM)",
    "revenue_growth": "營收成長率 (YoY)",
    "eps_growth": "每股盈餘成長率 (EPS Growth YoY)",
    "debt_to_equity": "負債權益比 (Debt to Equity Annual)"
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

def get_finnhub_json(endpoint, params):
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params["token"] = FINNHUB_API_KEY
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[Finnhub Error] Attempt {attempt + 1} for {endpoint}: {e}")
            time.sleep(2)
    logger.error(f"[Finnhub Error] Failed to fetch {endpoint} after 3 attempts")
    return {}

def get_quote(symbol, is_taiwan=False):
    if is_taiwan:
        try:
            data = get(symbol)
            if not data.get('success') or 'realtime' not in data:
                logger.error(f"Taiwan stock data fetch failed for {symbol}: {data.get('rtmessage', 'No data')}")
                return {}
            rt = data['realtime']
            current = rt['latest_trade_price'] if rt['latest_trade_price'] and rt['latest_trade_price'] != '--' else None
            prev = rt.get('previous_close') if rt.get('previous_close') and rt.get('previous_close') != '--' else None
            quote = {
                'current_price': round(float(current), 4) if current else 'N/A',
                'open': round(float(rt['open']), 4) if rt['open'] and rt['open'] != '--' else 'N/A',
                'high': round(float(rt['high']), 4) if rt['high'] and rt['high'] != '--' else 'N/A',
                'low': round(float(rt['low']), 4) if rt['low'] and rt['low'] != '--' else 'N/A',
                'previous_close': round(float(prev), 4) if prev else 'N/A',
                'daily_change': round((float(current) - float(prev)) / float(prev) * 100, 4) if current and prev and float(prev) != 0 else 'N/A',
                'volume': int(rt.get('accumulate_trade_volume', 0)) if rt.get('accumulate_trade_volume') and rt.get('accumulate_trade_volume') != '--' else 'N/A'
            }
            return quote
        except Exception as e:
            logger.error(f"Error fetching Taiwan quote for {symbol}: {e}")
            return {}
    else:
        try:
            data = get_finnhub_json("quote", {"symbol": symbol})
            if not data or 'c' not in data:
                logger.error(f"Finnhub quote fetch failed for {symbol}: {data}")
                return {}
            return {
                'current_price': round(data.get('c', 'N/A'), 4),
                'open': round(data.get('o', 'N/A'), 4),
                'high': round(data.get('h', 'N/A'), 4),
                'low': round(data.get('l', 'N/A'), 4),
                'previous_close': round(data.get('pc', 'N/A'), 4),
                'daily_change': round(data.get('dp', 'N/A'), 4),
                'volume': int(data.get('v', 'N/A')) if data.get('v') and data.get('v') != 'N/A' else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error fetching Finnhub quote for {symbol}: {e}")
            return {}

def get_metrics(symbol, is_taiwan=False):
    try:
        if is_taiwan:
            symbol = f"{symbol}.TW"
        metrics = get_finnhub_json("stock/metric", {"symbol": symbol, "metric": "all"}).get("metric", {})
        if not metrics:
            logger.warning(f"No metrics data for {symbol}")
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics for {symbol}: {e}")
        return {}

def filter_metrics(metrics):
    filtered = {}
    metric_map = {
        "peTTM": "pe_ratio",
        "pb": "pb_ratio",
        "roeTTM": "roe_ttm",
        "roaTTM": "roa_ttm",
        "grossMarginTTM": "gross_margin_ttm",
        "revenueGrowthTTMYoy": "revenue_growth",
        "epsGrowthTTMYoy": "eps_growth",
        "debtToEquityAnnual": "debt_to_equity"
    }
    for original_key, new_key in metric_map.items():
        v = metrics.get(original_key)
        if v is not None:
            try:
                v = float(v)
                if "growth" in new_key or "margin" in new_key or "roe" in new_key or "roa" in new_key:
                    filtered[new_key] = f"{v:.2f}%"
                else:
                    filtered[new_key] = round(v, 4)
            except:
                filtered[new_key] = str(v)
    return filtered

def get_recent_news(symbol, is_taiwan=False):
    try:
        if is_taiwan:
            symbol = f"{symbol}.TW"
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        past = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        news = get_finnhub_json("company-news", {"symbol": symbol, "from": past, "to": today})
        if not isinstance(news, list):
            logger.warning(f"No news data for {symbol}")
            return []
        news = sorted(news, key=lambda x: x.get("datetime", 0), reverse=True)[:10]
        for n in news:
            try:
                n["datetime"] = datetime.datetime.utcfromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M")
            except:
                n["datetime"] = "未知時間"
        return news
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []

def get_company_profile(symbol, is_taiwan=False):
    try:
        if is_taiwan:
            symbol = f"{symbol}.TW"
        profile = get_finnhub_json("stock/profile2", {"symbol": symbol})
        if not profile:
            logger.warning(f"No profile data for {symbol}")
        return profile
    except Exception as e:
        logger.error(f"Error fetching company profile for {symbol}: {e}")
        return {}

def calculate_rsi(series, period=14):
    try:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rs = rs.replace([float('inf'), -float('inf')], 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 'N/A'

def get_historical_data(symbol, is_taiwan=False):
    try:
        if is_taiwan:
            stock = Stock(symbol)
            today = datetime.datetime.today()
            before = today - datetime.timedelta(days=365)
            stock.fetch_from(before.year, before.month)
            if not stock.data:
                logger.warning(f"No historical data for Taiwan stock {symbol}")
                return pd.DataFrame(), {}
            data_dict = {
                'Date': stock.date,
                'Open': stock.open,
                'High': stock.high,
                'Low': stock.low,
                'Close': stock.close,
                'Volume': stock.capacity
            }
            df = pd.DataFrame(data_dict)
            df.set_index('Date', inplace=True)
            df = df.dropna(subset=['Close'])
        else:
            df = pd.DataFrame()
            for attempt in range(3):
                try:
                    df = yf.download(symbol, period="1y", progress=False)
                    if not df.empty:
                        break
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"[YF Historical Error] Attempt {attempt + 1} for {symbol}: {e}")
                    time.sleep(2)
            if df.empty:
                logger.warning(f"No historical data for {symbol}")
                return pd.DataFrame(), {}
        
        window_50 = min(50, len(df))
        ma50 = df['Close'].rolling(window=window_50).mean().iloc[-1] if len(df) >= 1 else 'N/A'
        rsi = calculate_rsi(df['Close'])
        ema12 = df['Close'].ewm(span=12, adjust=False).mean().iloc[-1] if len(df) >= 12 else 'N/A'
        ema26 = df['Close'].ewm(span=26, adjust=False).mean().iloc[-1] if len(df) >= 26 else 'N/A'
        macd = ema12 - ema26 if pd.notnull(ema12) and pd.notnull(ema26) else 'N/A'
        tail_20 = min(20, len(df))
        support = df['Low'].tail(tail_20).min() if len(df) >= 1 else 'N/A'
        resistance = df['High'].tail(tail_20).max() if len(df) >= 1 else 'N/A'
        volume = df['Volume'].iloc[-1] if 'Volume' in df.columns and len(df) > 0 else 'N/A'
        technical = {
            'ma50': round(float(ma50), 2) if pd.notnull(ma50) else 'N/A',
            'rsi': round(float(rsi), 2) if pd.notnull(rsi) else 'N/A',
            'macd': round(float(macd), 2) if pd.notnull(macd) else 'N/A',
            'support': round(float(support), 2) if pd.notnull(support) else 'N/A',
            'resistance': round(float(resistance), 2) if pd.notnull(resistance) else 'N/A',
            'volume': int(volume) if pd.notnull(volume) and volume != 'N/A' else 'N/A'
        }
        return df, technical
    except Exception as e:
        logger.error(f"Error in get_historical_data for {symbol}: {e}")
        return pd.DataFrame(), {}

def get_plot_html(df, symbol, currency="TWD"):
    try:
        if df.empty or 'Close' not in df.columns:
            return "<p class='text-danger'>📊 無法取得股價趨勢圖 / Unable to fetch price trend chart</p>"
        df_plot = df.tail(7)
        dates = df_plot.index.strftime('%Y-%m-%d').tolist()
        closes = df_plot['Close'].round(2).tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=closes, mode='lines+markers', name='Close Price'))
        fig.update_layout(
            title=f"{symbol} 最近7日收盤價 / 7-Day Closing Price Trend",
            xaxis_title="日期 / Date",
            yaxis_title=f"收盤價 ({currency})",
            template="plotly_white",
            height=400
        )
        return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height="400px", default_width="100%")
    except Exception as e:
        logger.error(f"Error generating plot for {symbol}: {e}")
        return "<p class='text-danger'>📊 無法生成股價趨勢圖 / Unable to generate price trend chart</p>"

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
            result["error"] = f"已達 {current_tier_name} 等級請求上限，請升級方案 / Request limit reached for {current_tier_name}, please upgrade your plan"
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        symbol = request.form.get("symbol", "").strip().upper()
        if not symbol:
            result["error"] = "請輸入股票代號 / Please enter a stock symbol"
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        try:
            session["request_count"] = request_count + 1
            # Determine if symbol is Taiwan stock (4-digit numeric)
            is_taiwan = len(symbol) == 4 and symbol.isdigit()
            currency = "TWD" if is_taiwan else "USD"

            # Fetch quote
            quote = get_quote(symbol, is_taiwan)
            if not quote or all(v == 'N/A' for v in quote.values()):
                result["error"] = f"無法取得 {symbol} 的即時報價資料 / Unable to fetch quote data for {symbol}"
                logger.error(f"No valid quote data for {symbol}")
                return render_template("index.html", result=result, symbol_input=symbol,
                                       tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                       stripe_mode=STRIPE_MODE, request_count=request_count,
                                       current_tier_name=current_tier_name, current_limit=current_limit)

            # Fetch other data
            metrics = filter_metrics(get_metrics(symbol, is_taiwan))
            news = get_recent_news(symbol, is_taiwan)
            profile = get_company_profile(symbol, is_taiwan)
            industry_en = profile.get("finnhubIndustry", "Unknown")
            industry_zh = industry_mapping.get(industry_en, "未知")
            df, technical = get_historical_data(symbol, is_taiwan)
            quote['volume'] = technical.get('volume', quote.get('volume', 'N/A'))
            plot_html = get_plot_html(df, symbol, currency)

            # Best Four Point analysis for Taiwan stocks
            bfp_signal = "無明確信號 / No clear signal"
            if is_taiwan:
                try:
                    stock = Stock(symbol)
                    if stock.data:
                        bfp = BestFourPoint(stock)
                        best = bfp.best_four_point()
                        if best:
                            bfp_signal = f"買入信號: {best[1]} / Buy signal: {best[1]}" if best[0] else f"賣出信號: {best[1]} / Sell signal: {best[1]}"
                    else:
                        logger.warning(f"No stock data for BestFourPoint analysis for {symbol}")
                except Exception as e:
                    logger.error(f"Error in BestFourPoint analysis for {symbol}: {e}")
                    bfp_signal = "無法計算最佳四點信號 / Unable to calculate Best Four Point signal"

            # Prepare prompt for GPT analysis
            technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items() if v != 'N/A')
            prompt = (
                f"請根據以下資訊產出中英文雙語股票分析: "
                f"股票代號: {symbol}, "
                f"目前價格: {quote.get('current_price', 'N/A')} {currency}, "
                f"產業分類: {industry_zh} ({industry_en}), "
                f"財務指標: {metrics}, "
                f"技術指標: {technical_str}, "
                f"最佳四點信號: {bfp_signal}. "
                f"請提供買入/賣出/持有建議."
            )
            try:
                chat_response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一位中英雙語金融分析助理，中英文內容完全對等。"
                                "請以JSON格式回應: "
                                "{'recommendation': 'buy' or 'sell' or 'hold', "
                                "'rationale': '中文 rationale\\nEnglish rationale', "
                                "'risk': '中文 risk\\nEnglish risk', "
                                "'summary': '中文 summary\\nEnglish summary'}"
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=999,
                    temperature=0.6,
                    response_format={"type": "json_object"}
                )
                gpt_analysis = json.loads(chat_response['choices'][0]['message']['content'])
            except Exception as e:
                logger.error(f"OpenAI API error for {symbol}: {e}")
                gpt_analysis = {
                    'summary': (
                        f"無法生成分析，請稍後重試 / Failed to generate analysis, please try again later\n\n"
                        f"---\n\n*以上分析僅供參考，投資有風險 / The above analysis is for reference only, investment carries risks*"
                    )
                }

            result = {
                "symbol": symbol,
                "quote": {k: v for k, v in quote.items() if v != 'N/A'},
                "industry_en": industry_en,
                "industry_zh": industry_zh,
                "metrics": metrics,
                "news": news,
                "gpt_analysis": gpt_analysis,
                "plot_html": plot_html,
                "technical": {k: v for k, v in technical.items() if v != 'N/A'},
                "is_taiwan": is_taiwan,
                "currency": currency,
                "bfp_signal": bfp_signal
            }
        except Exception as e:
            result = {"error": f"無法取得 {symbol} 的股票資料 / Unable to fetch data for {symbol}: {str(e)}"}
            logger.error(f"Processing error for symbol {symbol}: {e}")

    return render_template(
        "index.html",
        result=result,
        symbol_input=symbol,
        QUOTE_FIELDS=QUOTE_FIELDS,
        METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
        tiers=PRICING_TIERS,
        stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
        stripe_mode=STRIPE_MODE,
        request_count=request_count,
        current_tier_name=current_tier_name,
        current_limit=current_limit
    )

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
