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
    raise RuntimeError(f"❌ Stripe keys for mode '{STRIPE_MODE}' not set")
stripe.api_key = STRIPE_SECRET_KEY
openai.api_key = OPENAI_API_KEY

# ------------------ Logger setup ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Initialize Flask ------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

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
# Stock aliases for news queries
STOCK_ALIASES = {
    "2330": ["TSMC", "Taiwan Semiconductor", "台積電"],
    "2382": ["Quanta Computer", "廣達電腦"]
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

def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return 0
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.rolling(window=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_quote(symbol):
    try:
        if symbol not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return {}
        data = twrealtime.get(symbol)
        if not data.get('success'):
            logger.warning(f"No real-time data for {symbol}")
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
        logger.error(f"Error fetching quote for {symbol}: {e}")
        return {}

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
            logger.warning(f"No historical data for {symbol}")
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
        if symbol not in twcodes:
            logger.warning(f"Symbol {symbol} not found in twcodes")
            return {'name': STOCK_ALIASES.get(symbol, ['Unknown'])[0], 'group': '未知'}
        code_info = twcodes[symbol]
        return {
            'name': code_info.name,
            'group': code_info.group
        }
    except Exception as e:
        logger.error(f"Error fetching company profile for {symbol}: {e}")
        return {'name': STOCK_ALIASES.get(symbol, ['Unknown'])[0], 'group': '未知'}

def get_twse_news(symbol, company_name, limit=5):
    try:
        url = "https://www.twse.com.tw/en/news/newsList"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        news = []
        aliases = STOCK_ALIASES.get(symbol, [company_name])
        for item in soup.select('table tbody tr')[:limit * 2]:
            title_elem = item.select_one('td:nth-child(2) a')
            date_elem = item.select_one('td:nth-child(1)')
            if not title_elem or not date_elem:
                continue
            title = title_elem.text.strip()
            if any(alias.lower() in title.lower() for alias in aliases + [symbol]):
                news.append({
                    'title': title,
                    'url': 'https://www.twse.com.tw' + title_elem.get('href', '#'),
                    'published_at': date_elem.text.strip(),
                    'source': 'TWSE'
                })
        logger.info(f"Fetched {len(news)} TWSE news articles for {symbol}: {[article['title'] for article in news]}")
        return news[:limit]
    except Exception as e:
        logger.error(f"Error fetching TWSE news for {symbol}: {e}")
        return []

def get_cna_news(symbol, company_name, limit=5):
    try:
        aliases = STOCK_ALIASES.get(symbol, [company_name])
        query = '+'.join(aliases)
        url = f"https://www.cna.com.tw/search/hynews.aspx?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        news = []
        for item in soup.select('.mainList li')[:limit * 2]:
            title_elem = item.select_one('h2')
            date_elem = item.select_one('.date')
            url_elem = item.select_one('a')
            if title_elem and date_elem and url_elem:
                title = title_elem.text.strip()
                if any(alias.lower() in title.lower() for alias in aliases + [symbol]):
                    news.append({
                        'title': title,
                        'url': url_elem.get('href', '#'),
                        'published_at': date_elem.text.strip(),
                        'source': 'Central News Agency'
                    })
        logger.info(f"Fetched {len(news)} CNA news articles for {symbol}: {[article['title'] for article in news]}")
        return news[:limit]
    except Exception as e:
        logger.error(f"Error fetching CNA news for {symbol}: {e}")
        return []

def get_openai_news_summary(symbol, company_name, articles, limit=5):
    try:
        aliases = STOCK_ALIASES.get(symbol, [company_name])
        prompt = f"""
You are a financial news analyst. Given a list of news articles, summarize the most relevant ones for the stock {symbol} ({company_name}). Focus on articles from today or the past 7 days that directly relate to the stock's performance, corporate actions, or market impact. Return up to {limit} summaries in JSON format with fields: title, summary, source, published_at, url. If no articles are relevant, return an empty list. Articles:
{json.dumps(articles, ensure_ascii=False)}
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        summaries = json.loads(response.choices[0].message.content.strip())
        logger.info(f"OpenAI summarized {len(summaries)} news articles for {symbol}")
        return summaries[:limit]
    except Exception as e:
        logger.error(f"Error summarizing news with OpenAI for {symbol}: {e}")
        return []

def get_stock_news(symbol, company_name, limit=5):
    try:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        aliases = STOCK_ALIASES.get(symbol, [company_name])
        query = f"{' OR '.join(aliases + [symbol])}"
        logger.info(f"Sending NewsAPI query: {query} from {week_ago}")
        params = {
            'q': query,
            'from': week_ago,
            'sortBy': 'relevancy',
            'apiKey': NEWSAPI_KEY
        }
        response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"NewsAPI response status: {data.get('status')} | Total results: {data.get('totalResults', 0)}")
        articles = []
        for article in data.get('articles', [])[:limit * 2]:
            title = article.get('title', 'N/A')
            if any(alias.lower() in title.lower() for alias in aliases + [symbol]):
                articles.append({
                    'title': title,
                    'url': article.get('url', '#'),
                    'published_at': article.get('publishedAt', 'Unknown time'),
                    'source': article.get('source', {}).get('name', 'Unknown')
                })
        logger.info(f"Fetched {len(articles)} NewsAPI articles for {symbol}: {[article['title'] for article in articles]}")
        
        # Fetch TWSE and CNA news
        twse_news = get_twse_news(symbol, company_name, limit)
        cna_news = get_cna_news(symbol, company_name, limit)
        all_articles = articles + twse_news + cna_news
        
        # Use OpenAI to summarize and filter relevant news
        if all_articles:
            news = get_openai_news_summary(symbol, company_name, all_articles, limit)
        else:
            news = []
        
        # Fallback to NewsAPI query if no summarized results
        if not news:
            logger.info(f"No relevant news from OpenAI for {symbol}; trying broader NewsAPI query")
            params['q'] = f"{symbol} stock OR {company_name} stock"
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fallback NewsAPI response status: {data.get('status')} | Total results: {data.get('totalResults', 0)}")
            for article in data.get('articles', [])[:limit]:
                news.append({
                    'title': article.get('title', 'N/A'),
                    'url': article.get('url', '#'),
                    'published_at': article.get('publishedAt', 'Unknown time'),
                    'source': article.get('source', {}).get('name', 'Unknown')
                })
        logger.info(f"Final news count for {symbol}: {len(news)}")
        return news[:limit]
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []

def get_gpt_analysis(symbol, quote, technical, metrics, news):
    try:
        news_titles = [item['title'] for item in news]
        prompt = f"""
You are a financial analyst. Analyze the stock {symbol} based on the following data:
- Quote: {quote}
- Technical Indicators: {technical}
- Financial Metrics: {metrics}
- Recent News Titles: {news_titles}
Provide a concise analysis with:
1. Recommendation: "buy", "sell", or "hold"
2. Rationale: Explain why in 2-3 sentences
3. Risk Assessment: Low, Moderate, or High with a brief explanation
4. Summary: A 1-2 sentence summary of the stock's current status
Return the analysis in JSON format.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        analysis = json.loads(response.choices[0].message.content.strip())
        return analysis
    except Exception as e:
        logger.error(f"Error getting GPT analysis for {symbol}: {e}")
        return {
            'recommendation': None,
            'rationale': '無法取得 AI 回應 | Unable to obtain AI response',
            'risk': '未知 | Unknown',
            'summary': '請稍後重試 | Please try again later'
        }

def create_plot(df):
    if df.empty:
        return ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='#4B0082')))
    fig.update_layout(
        title='Stock Price Trend',
        xaxis_title='Date',
        yaxis_title='Price (TWD)',
        template='plotly_white',
        hovermode='x unified'
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# ------------------ Routes ------------------
@app.route('/')
def index():
    current_tier_name = session.get('current_tier_name', 'Free')
    request_count = int(session.get('request_count', 0))
    current_limit = next((tier['limit'] for tier in PRICING_TIERS if tier['name'] == current_tier_name), 50)
    symbol_input = request.form.get('symbol', '').strip().upper()
    result = {}
    
    if request.method == 'POST' and symbol_input:
        if request_count >= current_limit:
            result['error'] = f"已達請求上限 ({current_limit})，請升級方案或重置計數 | Request limit ({current_limit}) reached, please upgrade plan or reset count"
        elif not symbol_input.isdigit() or len(symbol_input) != 4:
            result['error'] = "請輸入有效的4位數字台股代號 (如 2330) | Please enter a valid 4-digit Taiwan stock symbol (e.g., 2330)"
        else:
            session['request_count'] = request_count + 1
            profile = get_company_profile(symbol_input)
            quote = get_quote(symbol_input)
            df, technical = get_historical_data(symbol_input)
            metrics = {key: 'N/A' for key in METRIC_NAMES_ZH_EN}
            news = get_stock_news(symbol_input, profile.get('name', 'Unknown'))
            gpt_analysis = get_gpt_analysis(symbol_input, quote, technical, metrics, news)
            plot_html = create_plot(df)
            result = {
                'symbol': symbol_input,
                'profile': profile,
                'industry_zh': industry_mapping.get(profile.get('group', 'Unknown'), '未知'),
                'industry_en': profile.get('group', 'Unknown'),
                'quote': quote,
                'technical': technical,
                'metrics': metrics,
                'gpt_analysis': gpt_analysis,
                'plot_html': plot_html,
                'news': news,
                'bfp_signal': TwBestFourPoint(TwStock(symbol_input)).best_four_point() if symbol_input in twcodes else '無明確信號 | No clear signal'
            }
    
    return render_template('index.html',
                           current_tier_name=current_tier_name,
                           request_count=request_count,
                           current_limit=current_limit,
                           symbol_input=symbol_input,
                           result=result,
                           QUOTE_FIELDS=QUOTE_FIELDS,
                           METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
                           tiers=PRICING_TIERS)

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    tier_name = request.form.get('tier')
    if not tier_name or tier_name not in [tier['name'] for tier in PRICING_TIERS]:
        return jsonify({'error': 'Invalid tier selected'}), 400
    price_id = STRIPE_PRICE_IDS.get(tier_name)
    if not validate_price_id(price_id, tier_name):
        return jsonify({'error': f'No price ID configured for {tier_name}'}), 400
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=url_for('index', _external=True) + '?success=true',
            cancel_url=url_for('index', _external=True) + '?canceled=true',
        )
        session['current_tier_name'] = tier_name
        session['request_count'] = 0
        return jsonify({'url': checkout_session.url})
    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        return jsonify({'error': 'Failed to create checkout session'}), 500

@app.route('/reset', methods=['POST'])
def reset():
    password = request.form.get('password')
    if password != 'RESET123':
        flash('無效的重置密碼 | Invalid reset password')
        return redirect(url_for('index'))
    session['request_count'] = 0
    flash('請求計數已重置 | Request count reset successfully')
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
