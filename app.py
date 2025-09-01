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
import re  # Added for input validation

# ---------- Safe casting helpers ----------
# (Unchanged, as they are robust)

# ---------- Basic checks & app setup ----------
# Ensure NumPy compatibility (updated to warn instead of raise)
if not np.__version__.startswith("1."):
    logger.warning(f"NumPy {np.__version__} may be incompatible — recommend numpy<2.0")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Require SECRET_KEY
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set")
app.config["SESSION_PERMANENT"] = False

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Stripe config ----------
STRIPE_TEST_SECRET_KEY = os.getenv("STRIPE_TEST_SECRET_KEY")
STRIPE_TEST_PUBLISHABLE_KEY = os.getenv("STRIPE_TEST_PUBLISHABLE_KEY")
STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY")
STRIPE_LIVE_PUBLISHABLE_KEY = os.getenv("STRIPE_LIVE_PUBLISHABLE_KEY")
STRIPE_MODE = os.getenv("STRIPE_MODE", "test").lower()
STRIPE_PRICE_IDS = {
    "Free": os.getenv("STRIPE_PRICE_TIER0"),
    "Basic": os.getenv("STRIPE_PRICE_TIER1"),
    "Pro": os.getenv("STRIPE_PRICE_TIER2"),
    "Premium": os.getenv("STRIPE_PRICE_TIER3"),
    "Enterprise": os.getenv("STRIPE_PRICE_TIER4"),
}
ENDPOINT_SECRET = os.getenv("STRIPE_ENDPOINT_SECRET")
if not ENDPOINT_SECRET:
    logger.warning("STRIPE_ENDPOINT_SECRET not set — webhook verification disabled (not secure for production)")

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
    raise RuntimeError("OpenAI API key not set in environment")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- In-memory storage & constants ----------
user_data = {}  # Consider replacing with a database in production

# (QUOTE_FIELDS, METRIC_NAMES_ZH_EN, PRICING_TIERS, STATIC_STOCK_NAMES, industry_mapping unchanged)

# Finnhub
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    logger.warning("Finnhub API key not set — some features will degrade (names, metrics, news).")

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
DataTuple = namedtuple(  # Renamed for PEP 8 compliance
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
        if not date_str or not isinstance(date_str, str):
            raise ValueError("Invalid or empty date string")
        # Handle various date formats
        match = re.match(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", date_str)
        if not match:
            raise ValueError(f"Unexpected date format: {date_str}")
        try:
            year = int(match.group(1)) + 1911
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            return f"{year}/{month}/{day}"
        except Exception as e:
            logger.error(f"Date conversion error for '{date_str}': {e}")
            raise

    def _make_datatuple(self, data):
        raise NotImplementedError

class TWSEFetcher(BaseFetcher):
    # (Unchanged, as it’s robust)

class TPEXFetcher(BaseFetcher):
    # (Unchanged, as it’s robust)

# ---------- Stock wrapper ----------
class Stock:
    # (Unchanged, but consider adding caching for fetch operations)

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
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            response = r.json()
            logger.info(f"Finnhub success: endpoint={endpoint} params={params}")
            return response
        except requests.HTTPError as e:
            status = getattr(r, "status_code", "N/A")
            logger.error(f"Finnhub HTTP error for {endpoint} (status {status}): {e}")  # Upgraded to error
            time.sleep(2)
        except Exception as e:
            logger.error(f"Finnhub request attempt {attempt+1} failed for {endpoint}: {e}")  # Upgraded to error
            time.sleep(2)
    logger.error(f"Finnhub API failed after retries for {endpoint}")
    return {}

def get_stock_name(symbol: str) -> str:
    logger.info(f"Fetching stock name for: {symbol}")
    if not re.match(r"^\d{4}$", symbol):  # Validate symbol
        logger.error(f"Invalid stock symbol: {symbol}")
        return "Invalid Symbol"
    if symbol in STATIC_STOCK_NAMES:
        return STATIC_STOCK_NAMES[symbol]
    if codes and symbol in codes:
        try:
            return codes[symbol].name
        except Exception:
            logger.warning("twstock.codes lookup failed — falling back to Finnhub/static.")
    profile = get_finnhub_json("stock/profile2", {"symbol": f"{symbol}.TW"})
    return profile.get("name", "Unknown")

# (get_quote, get_metrics, filter_metrics, get_company_profile unchanged)

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
        prices = [p for p in stock.close if isinstance(p, (int, float)) and not pd.isna(p)]
        if len(prices) < 10:  # Require minimum data points
            logger.warning(f"Insufficient data for technical analysis: {symbol}")
            return df, {"ma50": "N/A", "support": "N/A", "resistance": "N/A", "volume": "N/A"}
        if len(prices) >= 50:
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
    if df.empty or len(df) < 10:  # Require minimum data points
        logger.warning(f"Insufficient data to plot for {symbol}")
        return ""
    try:
        if "date" not in df.columns or "close" not in df.columns:
            logger.error(f"DataFrame missing required columns for {symbol}")
            return ""
        df_clean = df.dropna(subset=["date", "close"])  # Remove NaNs
        if df_clean.empty:
            logger.warning(f"No valid data to plot for {symbol}")
            return ""
        fig = px.line(df_clean, x="date", y="close", title=f"{symbol} Stock Price")
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (TWD)")
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"get_plot_html error for {symbol}: {e}")
        return ""

# (get_recent_news unchanged)

def call_openai_for_analysis(prompt: str) -> dict:
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
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or not all(k in parsed for k in ["recommendation", "rationale", "risk", "summary"]):
                logger.warning("OpenAI response missing required fields")
                raise ValueError("Invalid response structure")
            return parsed
        except Exception as e:
            logger.warning(f"OpenAI response parse failed: {e}; content: {content}")
            return {
                "recommendation": "hold",
                "rationale": "無法解析 OpenAI 回應，採用後備邏輯。| Failed to parse OpenAI response, using fallback.",
                "risk": "中等風險，請保持謹慎。| Moderate risk, use caution.",
                "summary": "OpenAI 回傳不可解析內容，採用保守建議。| OpenAI returned unparseable content; conservative recommendation applied.",
            }
    except (AuthenticationError, RateLimitError, APIError) as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "recommendation": "hold",
            "rationale": f"OpenAI API 錯誤: {str(e)}。採用後備邏輯。| OpenAI API error: {str(e)}. Using fallback.",
            "risk": "中等風險，監控市場動態。| Moderate risk, monitor market.",
            "summary": "OpenAI API 發生錯誤，採用後備建議。| OpenAI API error, using fallback recommendation.",
        }
    except Exception as e:
        logger.error(f"Unexpected OpenAI error: {e}")
        return {
            "recommendation": "hold",
            "rationale": f"分析失敗: {str(e)}，採用後備邏輯。| Analysis failed: {str(e)}, using fallback.",
            "risk": "中等風險，監控市場動態。| Moderate risk, monitor market.",
            "summary": "分析失敗，採用後備建議。| Analysis failed, using fallback recommendation.",
        }

# (get_stock_data unchanged, but consider caching results)

# ---------- Flask routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    symbol_input = ""
    result = {}
    user_id = request.remote_addr or "unknown"

    current_tier_index = safe_int(session.get("paid_tier", 0), 0)
    current_tier_index = min(max(current_tier_index, 0), len(PRICING_TIERS) - 1)
    current_tier = PRICING_TIERS[current_tier_index]
    current_tier_name = current_tier["name"]

    # Synchronize session and user_data
    request_count = safe_int(session.get("request_count", user_data.get(f"user:{user_id}:request_count", 0)), 0)
    user_data[f"user:{user_id}:request_count"] = request_count  # Sync to user_data
    current_limit = safe_int(current_tier.get("limit", 0), 0)

    if request.method == "POST":
        symbol = (request.form.get("symbol") or "").strip()
        symbol_input = symbol
        if not re.match(r"^\d{4}$", symbol):
            result = {"error": "股票代號必須為4位數字 | Stock ID must be a 4-digit number"}
        elif request_count >= current_limit:
            result = {"error": f"已達到 {current_tier_name} 方案的請求限制 ({current_limit}) | Request limit reached for {current_tier_name} tier ({current_limit})"}
        else:
            result = get_stock_data(symbol)
            if "error" not in result:
                request_count += 1
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
    tier_name = request.form.get("tier")
    tier = next((t for t in PRICING_TIERS if t["name"] == tier_name), None)
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
        logger.error(f"Stripe checkout creation failed: {e}")  # Upgraded to error
        return jsonify({"error": f"無法創建結帳會話: {str(e)} | Failed to create checkout session: {str(e)}"}), 500

@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature")
    try:
        if not ENDPOINT_SECRET:
            raise ValueError("STRIPE_ENDPOINT_SECRET is not set")
        event = stripe.Webhook.construct_event(payload, sig_header, ENDPOINT_SECRET)
        if event.get("type") == "checkout.session.completed":
            stripe_session = event["data"]["object"]
            metadata = stripe_session.get("metadata", {})
            user_id = metadata.get("user_id")
            tier = metadata.get("tier")
            tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t["name"] == tier), None)
            if tier_index is not None and user_id:
                user_data[f"user:{user_id}:tier"] = tier
                user_data[f"user:{user_id}:request_count"] = 0
                logger.info("Webhook: Updated %s to %s tier", user_id, tier)
    except ValueError as e:
        logger.error(f"Webhook verification failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "success"})

@app.route("/reset", methods=["POST"])
def reset():
    password = request.form.get("password")
    reset_password = os.getenv("RESET_PASSWORD")
    if not reset_password:
        logger.error("RESET_PASSWORD not set")
        flash("❌ 重置功能未啟用 | Reset function not enabled.", "danger")
        return redirect(url_for("index"))
    if password == reset_password:
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
    app.run(host="0.0.0.0", port=safe_int(os.getenv("PORT", 8080), 8080), debug=False)  # Disable debug in production
