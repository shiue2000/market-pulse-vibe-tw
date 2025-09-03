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
from bs4 import BeautifulSoup

# ------------------ Load environment ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "327ab6e463624447901ecee80b7dcb0b")
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
    raise RuntimeError("❌ OPENAI_API_KEY not set in environment variables")
if not NEWSAPI_KEY:
    logger.warning("⚠️ NEWSAPI_KEY not set; news fetching may be limited")
# Set Stripe keys
if STRIPE_MODE == "live":
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY
if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
    raise RuntimeError(f"❌ Stripe keys for mode '{STRIPE_MODE}' not set in environment variables")
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

# Hard-coded symbol mappings for common cases (case-insensitive)
SYMBOL_MAPPINGS = {
    "台積電": "2330.TW",
    "tsmc": "2330.TW",
    "taiwan semiconductor manufacturing": "2330.TW",
    "2330": "2330.TW"
}

# ------------------ Helper functions ------------------
def validate_price_id(price_id, tier_name):
    return bool(price_id)

def resolve_symbol(input_str):
    input_str = input_str.strip().lower()  # Normalize input to lowercase
    if not input_str:
        return None
    logger.info(f"Resolving symbol: {input_str}")
    
    # Initialize cache if not present
    if "symbol_cache" not in session:
        session["symbol_cache"] = {}
    cache = session["symbol_cache"]

    # Check cache
    if input_str in cache:
        logger.info(f"Cache hit for {input_str}: {cache[input_str]}")
        return cache[input_str]

    # Check hard-coded mappings
    for key, symbol in SYMBOL_MAPPINGS.items():
        if input_str == key.lower():
            profile = get_company_profile(symbol.split('.')[0])
            if profile and profile.get('name'):
                cache[input_str] = symbol
                session["symbol_cache"] = cache
                logger.info(f"Resolved {input_str} to {symbol} via mapping")
                return symbol

    # Handle direct symbols with .TW or .TWO
    if '.' in input_str:
        parts = input_str.rsplit('.', 1)
        symbol = parts[0].upper() + '.' + parts[1].upper()
        if symbol.endswith('.TW') or symbol.endswith('.TWO'):
            profile = get_company_profile(symbol.split('.')[0])
            if profile and profile.get('name'):
                cache[input_str] = symbol
                session["symbol_cache"] = cache
                logger.info(f"Resolved {input_str} to {symbol} via direct symbol")
                return symbol

    # Handle numeric stock IDs
    if input_str.isdigit():
        for suffix in ['.TW', '.TWO']:
            symbol = input_str + suffix
            profile = get_company_profile(symbol.split('.')[0])
            if profile and profile.get('name'):
                cache[input_str] = symbol
                session["symbol_cache"] = cache
                logger.info(f"Resolved {input_str} to {symbol} via numeric ID")
                return symbol
        return None

    # Handle company names via twcodes search
    for code, info in twcodes.items():
        if input_str in info.name.lower() or input_str in code.lower():
            for suffix in ['.TW', '.TWO']:
                symbol = code + suffix
                profile = get_company_profile(code)
                if profile and profile.get('name'):
                    cache[input_str] = symbol
                    session["symbol_cache"] = cache
                    logger.info(f"Resolved {input_str} to {symbol} via twcodes search")
                    return symbol

    # Fallback to OpenAI
    try:
        prompt = (
            f"將以下輸入轉換為台灣股票代號（必須以 .TW 或 .TWO 結尾，例如 2330.TW）。"
            f"如果輸入是 '台積電'、'TSMC'、'tsmc' 或 '2330'，應回傳 '2330.TW'。輸入：{input_str}。僅回覆代號，例如 2330.TW。"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.2
        )
        suggested_symbol = response['choices'][0]['message']['content'].strip()
        if suggested_symbol.endswith('.TW') or suggested_symbol.endswith('.TWO'):
            profile = get_company_profile(suggested_symbol.split('.')[0])
            if profile and profile.get('name'):
                cache[input_str] = suggested_symbol
                session["symbol_cache"] = cache
                logger.info(f"Resolved {input_str} to {suggested_symbol} via OpenAI")
                return suggested_symbol
    except Exception as e:
        logger.warning(f"OpenAI symbol resolution error for {input_str}: {e}")

    logger.warning(f"Failed to resolve symbol: {input_str}")
    return None

def get_quote(symbol):
    try:
        code = symbol.split('.')[0]
        if code not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return {}
        data = twrealtime.get(code)
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
        # Fetch previous close from historical data
        stock = TwStock(code)
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
        logger.error(f"Error fetching quote for {symbol}: {e}")
        return {}

def get_historical_data(symbol):
    try:
        code = symbol.split('.')[0]
        if code not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return pd.DataFrame(), {}
        stock = TwStock(code)
        current_year = datetime.datetime.now().year
        stock.fetch_from(current_year - 1, 1)  # Fetch data from January of last year to now
        df = pd.DataFrame(stock.data)
        if df.empty:
            logger.warning(f"No historical data for symbol {symbol}")
            return pd.DataFrame(), {}
        df = df.rename(columns={'date': 'Date', 'capacity': 'Volume', 'turnover': 'Turnover', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'change': 'Change', 'transaction': 'Transaction'})
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
                'ma50': round(ma50, 2),
                'rsi': round(rsi, 2),
                'macd': round(macd, 2),
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'volume': volume
            }
        return df, technical
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame(), {}

def get_company_profile(symbol):
    try:
        code = symbol.split('.')[0]
        if code not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return {'name': 'N/A', 'group': '未知'}
        code_info = twcodes[code]
        return {
            'name': code_info.name,
            'group': code_info.group
        }
    except Exception as e:
        logger.error(f"Error fetching company profile for {symbol}: {e}")
        return {'name': 'N/A', 'group': '未知'}

def get_twse_news(symbol, company_name, limit=5):
    try:
        url = "https://www.twse.com.tw/en/announcement/list"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        news = []
        # Parse TWSE news (adjust selectors based on actual TWSE HTML structure)
        for item in soup.select('table tr')[:limit * 2]:  # Fetch more to filter relevant
            title_elem = item.select_one('td:nth-child(2) a')
            date_elem = item.select_one('td:nth-child(1)')
            if title_elem and date_elem:
                title = title_elem.text.strip()
                # Filter for company_name or symbol
                if company_name.lower() in title.lower() or symbol.lower() in title.lower():
                    news.append({
                        'title': title,
                        'url': 'https://www.twse.com.tw' + title_elem.get('href', '#'),
                        'published_at': date_elem.text.strip(),
                        'source': 'Taiwan Stock Exchange'
                    })
        logger.info(f"Fetched {len(news)} TWSE news articles for {symbol}: {[article['title'] for article in news]}")
        return news[:limit]
    except Exception as e:
        logger.error(f"Error fetching TWSE news for {symbol}: {e}")
        return []

def get_stock_news(symbol, company_name, limit=5):
    news = []
    if NEWSAPI_KEY:
        try:
            # Primary query with exact company name and symbol
            query = f"\"{company_name}\" OR \"{symbol}\""
            from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': NEWSAPI_KEY
            }
            logger.info(f"Sending NewsAPI query: {query} from {from_date}")
            response = requests.get("https://newsapi.org/v2/everything", params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"NewsAPI response status: {data.get('status')} | Total results: {data.get('totalResults', 0)}")
            if data.get('status') == 'ok':
                articles = data.get('articles', [])[:limit]
                news = [
                    {
                        'title': article.get('title', 'N/A'),
                        'url': article.get('url', '#'),
                        'published_at': article.get('publishedAt', 'N/A'),
                        'source': article.get('source', {}).get('name', 'Unknown')
                    }
                    for article in articles
                ]
                logger.info(f"Fetched {len(news)} NewsAPI articles for {symbol}: {[article['title'] for article in news]}")
            else:
                logger.warning(f"NewsAPI error: {data.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news for {symbol}: {e}")
    if not news:
        logger.info(f"No NewsAPI results for {symbol}; falling back to TWSE")
        news = get_twse_news(symbol, company_name, limit)
    if not news:
        logger.info(f"No TWSE results for {symbol}; trying broader NewsAPI query")
        try:
            params = {
                'q': f"{symbol} stock",
                'from': (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': NEWSAPI_KEY
            }
            logger.info(f"Sending fallback NewsAPI query: {params['q']}")
            response = requests.get("https://newsapi.org/v2/everything", params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])[:limit]
                news = [
                    {
                        'title': article.get('title', 'N/A'),
                        'url': article.get('url', '#'),
                        'published_at': article.get('publishedAt', 'N/A'),
                        'source': article.get('source', {}).get('name', 'Unknown')
                    }
                    for article in articles
                ]
                logger.info(f"Fallback query fetched {len(news)} NewsAPI articles for {symbol}: {[article['title'] for article in news]}")
        except Exception as e:
            logger.error(f"Error fetching fallback NewsAPI news for {symbol}: {e}")
    return news

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_plot_html(df, symbol):
    if df.empty or 'Close' not in df.columns:
        logger.warning(f"No data to plot for {symbol}")
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
    symbol_input = ""
    symbol = ""
    current_tier_index = session.get("paid_tier", 0)
    current_tier = PRICING_TIERS[current_tier_index]
    request_count = session.get("request_count", 0)
    current_limit = current_tier["limit"]
    current_tier_name = current_tier["name"]
    if request.method == "POST":
        if request_count >= current_limit:
            result["error"] = f"已達 {current_tier_name} 等級請求上限，請升級方案"
            return render_template("index.html", result=result, symbol_input=symbol_input,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)
        symbol_input = request.form.get("symbol", "").strip()
        if not symbol_input:
            result["error"] = "請輸入股票代號、名稱或ID / Please enter a stock symbol, name, or ID"
            return render_template("index.html", result=result, symbol_input=symbol_input,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)
        symbol = resolve_symbol(symbol_input)
        if not symbol:
            result = {
                "error": f"無效的股票代號或名稱: {symbol_input} / Invalid stock symbol or name: {symbol_input}",
                "profile": {'name': 'N/A', 'group': '未知'},
                "news": []
            }
            return render_template("index.html", result=result, symbol_input=symbol_input,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)
        try:
            session["request_count"] = request_count + 1
            quote = get_quote(symbol)
            metrics = {}  # Skip, or use custom calculation if needed
            profile = get_company_profile(symbol.split('.')[0])
            company_name = profile.get('name', 'Unknown')
            news = get_stock_news(symbol, company_name)
            industry_zh = profile.get('group', '未知')
            industry_en = next((en for en, zh in industry_mapping.items() if zh == industry_zh), "Unknown")
            df, technical = get_historical_data(symbol)
            plot_html = get_plot_html(df, symbol)
            bfp_signal = "無明確信號 / No clear signal"
            try:
                stock = TwStock(symbol.split('.')[0])
                stock.fetch_31()  # Fetch recent data for BestFourPoint analysis
                bfp = TwBestFourPoint(stock)
                best = bfp.best_four_point()
                if best:
                    bfp_signal = f"買入信號: {best[1]}" if best[0] else f"賣出信號: {best[1]}"
            except Exception as e:
                logger.error(f"Error in BestFourPoint analysis for {symbol}: {e}")
            technical_str = ", ".join(f"{k.upper()}: {v}" for k, v in technical.items() if v != 'N/A')
            prompt = f"請根據以下資訊產出中英文雙語股票分析: 股票代號: {symbol}, 目前價格: {quote.get('current_price', 'N/A')}, 產業分類: {industry_zh} ({industry_en}), 財務指標: {metrics}, 技術指標: {technical_str}, 最佳四點信號: {bfp_signal}. 請提供買入/賣出/持有建議."
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
            result = {
                "symbol": symbol,
                "quote": quote,
                "industry_en": industry_en,
                "industry_zh": industry_zh,
                "metrics": metrics,
                "news": news,
                "gpt_analysis": gpt_analysis,
                "plot_html": plot_html,
                "technical": technical,
                "profile": profile,
                "bfp_signal": bfp_signal
            }
        except Exception as e:
            result = {
                "error": f"資料讀取錯誤: {e} / Data retrieval error: {e}",
                "profile": {'name': 'N/A', 'group': '未知'},
                "news": []
            }
            logger.error(f"Processing error for symbol {symbol}: {e}")
    return render_template("index.html",
                           result=result,
                           symbol_input=symbol_input,
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
