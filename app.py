# -*- coding: utf-8 -*-
import datetime
import requests
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import openai
import plotly.graph_objs as go
import stripe
import os
import logging
import time
import pandas as pd
import json
from twstock import Stock as TwStock, realtime as twrealtime, codes as twcodes
from twstock import BestFourPoint as TwBestFourPoint
from functools import lru_cache
import secrets
import re

# ------------------ Logger setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------ Load environment ------------------
def load_env_variable(var_name, required=True, default=None):
    """Load environment variable with validation."""
    value = os.getenv(var_name, default)
    if required and not value:
        logger.error(f"Missing environment variable: {var_name}")
        flash(f"⚠️ Server configuration error: {var_name} is missing.", "danger")
        raise RuntimeError(f"❌ {var_name} not set")
    return value

OPENAI_API_KEY = load_env_variable("OPENAI_API_KEY")
SECRET_KEY = load_env_variable("SECRET_KEY", default=secrets.token_hex(16))
STRIPE_MODE = load_env_variable("STRIPE_MODE", default="test").lower()
RESET_PASSWORD = load_env_variable("RESET_PASSWORD", default=secrets.token_hex(16))

# Stripe keys
STRIPE_TEST_SECRET_KEY = load_env_variable("STRIPE_TEST_SECRET_KEY", required=STRIPE_MODE == "test")
STRIPE_TEST_PUBLISHABLE_KEY = load_env_variable("STRIPE_TEST_PUBLISHABLE_KEY", required=STRIPE_MODE == "test")
STRIPE_LIVE_SECRET_KEY = load_env_variable("STRIPE_LIVE_SECRET_KEY", required=STRIPE_MODE == "live")
STRIPE_LIVE_PUBLISHABLE_KEY = load_env_variable("STRIPE_LIVE_PUBLISHABLE_KEY", required=STRIPE_MODE == "live")

# Stripe Price IDs
STRIPE_PRICE_IDS = {
    "Free": load_env_variable("STRIPE_PRICE_TIER0", required=False),
    "Tier 1": load_env_variable("STRIPE_PRICE_TIER1", required=False),
    "Tier 2": load_env_variable("STRIPE_PRICE_TIER2", required=False),
    "Tier 3": load_env_variable("STRIPE_PRICE_TIER3", required=False),
    "Tier 4": load_env_variable("STRIPE_PRICE_TIER4", required=False),
}

if STRIPE_MODE == "live":
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY

stripe.api_key = STRIPE_SECRET_KEY

# ------------------ Initialize Flask & OpenAI ------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(minutes=30)
)
openai.api_key = OPENAI_API_KEY

# Test API connectivity
def test_openai_api():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        logger.info("OpenAI API test successful")
        return True
    except Exception as e:
        logger.error(f"OpenAI API test failed: {e}")
        return False

def test_stripe_api():
    try:
        stripe.Price.list(limit=1)
        logger.info("Stripe API test successful")
        return True
    except Exception as e:
        logger.error(f"Stripe API test failed: {e}")
        return False

# Run API tests at startup
if not test_openai_api():
    logger.error("OpenAI API is not accessible. Check API key and permissions.")
    flash("⚠️ Analysis service is currently unavailable.", "danger")
if not test_stripe_api():
    logger.error("Stripe API is not accessible. Check keys and mode.")
    flash("⚠️ Payment system is currently unavailable.", "danger")

# ------------------ Stock app config ------------------
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
    """Validate Stripe price ID with API check."""
    if not price_id:
        return False
    try:
        stripe.Price.retrieve(price_id)
        return True
    except Exception as e:
        logger.error(f"Invalid Price ID for {tier_name}: {e}")
        return False

@lru_cache(maxsize=128)
def get_quote(symbol):
    try:
        if symbol not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return {}
        data = twrealtime.get(symbol)
        if not data.get('success'):
            logger.warning(f"No real-time data for symbol {symbol}")
            return {}
        rt = data['realtime']
        current_price = rt.get('latest_trade_price', 'N/A')
        quote = {
            'current_price': current_price,
            'open': rt.get('open', 'N/A'),
            'high': rt.get('high', 'N/A'),
            'low': rt.get('low', 'N/A'),
            'previous_close': 'N/A',
            'daily_change': 'N/A',
            'volume': rt.get('accumulate_trade_volume', 'N/A')
        }
        stock = TwStock(symbol)
        historical = stock.fetch_31()
        if historical:
            previous_close = historical[-1].close
            quote['previous_close'] = previous_close
            if current_price != 'N/A' and current_price != '-' and previous_close:
                try:
                    change = (float(current_price) - previous_close) / previous_close * 100
                    quote['daily_change'] = round(change, 2)
                except ValueError:
                    logger.warning(f"Unable to calculate daily change for {symbol}")
        return quote
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}", exc_info=True)
        return {}

@lru_cache(maxsize=128)
def get_historical_data(symbol):
    try:
        if symbol not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return pd.DataFrame(), {}
        stock = TwStock(symbol)
        current_year = datetime.datetime.now().year
        stock.fetch_from(current_year - 1, 1)
        df = pd.DataFrame(stock.data)
        if df.empty:
            logger.warning(f"No historical data for symbol {symbol}")
            return pd.DataFrame(), {}
        df = df.rename(columns={
            'date': 'Date', 'capacity': 'Volume', 'turnover': 'Turnover',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'change': 'Change', 'transaction': 'Transaction'
        })
        df.set_index('Date', inplace=True)
        technical = {}
        if not df.empty:
            window_50 = min(50, len(df))
            ma50 = df['Close'].rolling(window=window_50).mean().iloc[-1]
            rsi = calculate_rsi(df['Close'])
            ema12 = df['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = df['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
            macd = ema12 - ema26
            tail_20 = min(20, len(df))
            support = df['Low'].tail(tail_20).min()
            resistance = df['High'].tail(tail_20).max()
            volume = df['Volume'].iloc[-1]
            technical = {
                'ma50': round(ma50, 2) if pd.notna(ma50) else 'N/A',
                'rsi': round(rsi, 2) if pd.notna(rsi) else 'N/A',
                'macd': round(macd, 2) if pd.notna(macd) else 'N/A',
                'support': round(support, 2) if pd.notna(support) else 'N/A',
                'resistance': round(resistance, 2) if pd.notna(resistance) else 'N/A',
                'volume': volume
            }
        return df, technical
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame(), {}

@lru_cache(maxsize=128)
def get_company_profile(symbol):
    try:
        if symbol not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return {'name': 'N/A', 'group': '未知'}
        code_info = twcodes[symbol]
        return {
            'name': code_info.name,
            'group': code_info.group
        }
    except Exception as e:
        logger.error(f"Error fetching company profile for {symbol}: {e}", exc_info=True)
        return {'name': 'N/A', 'group': '未知'}

@lru_cache(maxsize=128)
def get_stock_news(symbol, company_name, limit=10):
    try:
        from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        prompt = f"""
        Fetch the top {limit} most relevant news articles about the company {company_name} (stock symbol: {symbol}) from the past 30 days.
        Return the results in JSON format with the following structure for each article:
        {{
            "title": "article title",
            "url": "article URL",
            "published_at": "publication date in YYYY-MM-DD format",
            "source": "source name"
        }}
        Ensure the articles are in English or Chinese and are directly related to the company or stock.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial news assistant capable of fetching and summarizing recent news articles. Provide accurate and relevant news data in JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        news_data = json.loads(response['choices'][0]['message']['content'])
        news = news_data.get('articles', [])[:limit]
        logger.info(f"Fetched {len(news)} news articles for {symbol}")
        return news
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}", exc_info=True)
        return []

def calculate_rsi(series, period=14):
    try:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if pd.notna(rsi.iloc[-1]) else 'N/A'
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 'N/A'

def get_plot_html(df, symbol):
    if df.empty or 'Close' not in df.columns:
        logger.warning(f"No data to plot for {symbol}")
        return "<p class='text-danger'>📊 無法取得股價趨勢圖 / No chart available</p>"
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
            result["error"] = f"已達 {current_tier_name} 等級請求上限，請升級方案 / Request limit reached for {current_tier_name} plan"
            return render_template("index.html", result=result, symbol_input=symbol,
                                 tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                 stripe_mode=STRIPE_MODE, request_count=request_count,
                                 current_tier_name=current_tier_name, current_limit=current_limit)
        symbol = request.form.get("symbol", "").strip().upper()
        if not symbol or not re.match(r"^[A-Z0-9]{1,10}$", symbol):
            result["error"] = "無效的股票代號 / Invalid stock symbol"
            return render_template("index.html", result=result, symbol_input=symbol,
                                 tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                 stripe_mode=STRIPE_MODE, request_count=request_count,
                                 current_tier_name=current_tier_name, current_limit=current_limit)
        if symbol not in twcodes:
            result = {
                "error": f"無效的股票代號: {symbol} / Invalid stock symbol: {symbol}",
                "profile": {'name': 'N/A', 'group': '未知'},
                "news": []
            }
            return render_template("index.html", result=result, symbol_input=symbol,
                                 tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                 stripe_mode=STRIPE_MODE, request_count=request_count,
                                 current_tier_name=current_tier_name, current_limit=current_limit)
        try:
            session["request_count"] = request_count + 1
            quote = get_quote(symbol)
            profile = get_company_profile(symbol)
            company_name = profile.get('name', 'Unknown')
            news = get_stock_news(symbol, company_name)
            industry_zh = profile.get('group', '未知')
            industry_en = next((en for en, zh in industry_mapping.items() if zh == industry_zh), "Unknown")
            df, technical = get_historical_data(symbol)
            plot_html = get_plot_html(df, symbol)
            bfp_signal = "無明確信號 / No clear signal"
            try:
                stock = TwStock(symbol)
                stock.fetch_31()
                bfp = TwBestFourPoint(stock)
                best = bfp.best_four_point()
                if best:
                    bfp_signal = f"買入信號: {best[1]}" if best[0] else f"賣出信號: {best[1]}"
            except Exception as e:
                logger.error(f"BestFourPoint analysis failed for {symbol}: {e}", exc_info=True)
            technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items() if v != 'N/A')
            prompt = f"請根據以下資訊產出中英文雙語股票分析: 股票代號: {symbol}, 目前價格: {quote.get('current_price', 'N/A')}, 產業分類: {industry_zh} ({industry_en}), 技術指標: {technical_str}, 最佳四點信號: {bfp_signal}. 請提供買入/賣出/持有建議."
            try:
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
                gpt_analysis = json.loads(chat_response['choices'][0]['message']['content'])
            except Exception as e:
                logger.error(f"OpenAI analysis failed for {symbol}: {e}", exc_info=True)
                gpt_analysis = {
                    "recommendation": "N/A",
                    "rationale": "分析失敗 / Analysis failed",
                    "risk": "N/A",
                    "summary": "N/A"
                }
            result = {
                "symbol": symbol,
                "quote": quote,
                "industry_en": industry_en,
                "industry_zh": industry_zh,
                "metrics": {},  # Placeholder for future metrics
                "news": news,
                "gpt_analysis": gpt_analysis,
                "plot_html": plot_html,
                "technical": technical,
                "profile": profile,
                "bfp_signal": bfp_signal
            }
        except Exception as e:
            logger.error(f"Processing error for {symbol}: {e}", exc_info=True)
            result = {
                "error": f"資料讀取錯誤: {e} / Data retrieval error: {e}",
                "profile": {'name': 'N/A', 'group': '未知'},
                "news": []
            }
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

@app.route("/details/<symbol>")
def stock_details(symbol):
    if not re.match(r"^[A-Z0-9]{1,10}$", symbol):
        flash("❌ Invalid stock symbol.", "danger")
        return redirect(url_for("index"))
    result = {}
    try:
        quote = get_quote(symbol)
        profile = get_company_profile(symbol)
        company_name = profile.get('name', 'Unknown')
        news = get_stock_news(symbol, company_name)
        df, technical = get_historical_data(symbol)
        plot_html = get_plot_html(df, symbol)
        bfp_signal = "無明確信號 / No clear signal"
        try:
            stock = TwStock(symbol)
            stock.fetch_31()
            bfp = TwBestFourPoint(stock)
            best = bfp.best_four_point()
            if best:
                bfp_signal = f"買入信號: {best[1]}" if best[0] else f"賣出信號: {best[1]}"
        except Exception as e:
            logger.error(f"BestFourPoint analysis failed for {symbol}: {e}", exc_info=True)
        result = {
            "symbol": symbol,
            "quote": quote,
            "news": news,
            "plot_html": plot_html,
            "technical": technical,
            "profile": profile,
            "bfp_signal": bfp_signal
        }
    except Exception as e:
        logger.error(f"Error fetching details for {symbol}: {e}", exc_info=True)
        result["error"] = f"資料讀取錯誤: {e} / Data retrieval error: {e}"
    return render_template(
        "details.html",
        result=result,
        QUOTE_FIELDS=QUOTE_FIELDS,
        METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN
    )

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
        logger.error(f"Unexpected Stripe error: {e}", exc_info=True)
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
    if password == RESET_PASSWORD:
        session["request_count"] = 0
        session["subscribed"] = False
        session["paid_tier"] = 0
        flash("✅ Counts reset.", "success")
        logger.info("Session counts reset successfully")
    else:
        flash("❌ Incorrect password.", "danger")
        logger.warning("Failed reset attempt")
    return redirect(url_for("index"))

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
