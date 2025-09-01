# app.py
import os
import time
import datetime
import logging
from functools import wraps, lru_cache

import requests
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

from flask import (
    Flask, request, render_template, jsonify, redirect, session, flash, url_for
)

# Optional ML deps (only used when available)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from peft import PeftModel
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# --- App & logging ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Stripe config ---
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

if STRIPE_MODE == "live":
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY

if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
    logger.warning(f"Stripe keys for mode '{STRIPE_MODE}' not fully configured. Checkout will fail until keys set.")

try:
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    STRIPE_INSTALLED = True
except Exception:
    STRIPE_INSTALLED = False
    logger.warning("stripe package not available. Stripe endpoints will be disabled.")

# --- Pricing & UI labels ---
PRICING_TIERS = [
    {"name": "Free", "limit": 50, "price": 0},
    {"name": "Basic", "limit": 200, "price": 5},
    {"name": "Pro", "limit": 500, "price": 10},
    {"name": "Premium", "limit": 1000, "price": 20},
    {"name": "Enterprise", "limit": 3000, "price": 50},
]

QUOTE_FIELDS = {
    "current_price": ("目前價格", "Current Price"),
    "daily_change": ("當日變化", "Daily Change (%)"),
    "volume": ("交易量", "Volume"),
    "open_price": ("開盤價", "Open Price"),
    "high_price": ("最高價", "High Price"),
    "low_price": ("最低價", "Low Price"),
    "prev_close": ("前日收盤價", "Previous Close")
}

METRIC_NAMES_ZH_EN = {
    "pe": "本益比 (PE TTM) | PE Ratio (TTM)",
    "pb": "股價淨值比 (PB) | PB Ratio",
    "revenue_growth": "營收成長率 (YoY) | Revenue Growth (YoY)",
    "eps_growth": "每股盈餘成長率 (YoY) | EPS Growth (YoY)",
}

# --- Finnhub ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# --- In-memory user storage & simple rate limiting ---
user_data = {}  # persistent-ish in-memory: not suitable for multi-process
RATE_LIMIT_WINDOW_SECONDS = 3600  # window per tier
ip_requests = {}  # { ip: {"count": int, "window_start": timestamp} }


def get_client_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr)


def check_rate_limit(ip: str, limit: int) -> bool:
    now = time.time()
    rec = ip_requests.get(ip)
    if not rec or now - rec["window_start"] > RATE_LIMIT_WINDOW_SECONDS:
        ip_requests[ip] = {"count": 0, "window_start": now}
        rec = ip_requests[ip]
    if rec["count"] >= limit:
        return False
    rec["count"] += 1
    return True


# --- Lazy model init for Llama (QLoRA) ---
MODEL_CONFIG = {
    "base": "DavidLanz/Llama3-tw-8B-Instruct",
    "adapter": "DavidLanz/llama3_8b_taiwan_stock_qlora",
    # Quant config defaults
    "use_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": False,
}
llama_pipeline = None
llama_tokenizer = None
llama_initialized = False


def initialize_llama_model():
    global llama_pipeline, llama_tokenizer, llama_initialized
    if llama_initialized:
        return
    llama_initialized = True  # mark to avoid repeated attempts
    if not TORCH_AVAILABLE:
        logger.warning("Torch or transformers not available - skipping Llama model init.")
        return

    try:
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        compute_dtype = getattr(torch, MODEL_CONFIG["bnb_4bit_compute_dtype"], torch.float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=MODEL_CONFIG["use_4bit"],
            bnb_4bit_quant_type=MODEL_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=MODEL_CONFIG["bnb_4bit_use_double_quant"],
        ) if torch.cuda.is_available() else None

        logger.info("Loading base model (may take time)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["base"],
            low_cpu_mem_usage=True,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map=device_map,
            quantization_config=bnb_config if bnb_config is not None else None,
        )

        model = PeftModel.from_pretrained(base_model, MODEL_CONFIG["adapter"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["base"], trust_remote_code=True)
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map if isinstance(device_map, dict) else None,
        )

        llama_pipeline = text_gen_pipeline
        llama_tokenizer = tokenizer
        logger.info("Llama model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Llama model: {e}")
        llama_pipeline, llama_tokenizer = None, None


# --- Utility: safe Finnhub call ---
def get_finnhub_json(endpoint: str, params: dict):
    if not FINNHUB_API_KEY:
        return {}
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params = params.copy()
    params["token"] = FINNHUB_API_KEY
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[Finnhub Error] {endpoint}: {e}")
            time.sleep(1)
    return {}


# --- Recent news (10 days) ---
def get_recent_news(symbol: str):
    try:
        today = datetime.datetime.utcnow()
        past = today - datetime.timedelta(days=10)
        news = get_finnhub_json(
            "company-news",
            {"symbol": f"{symbol}.TW", "from": past.strftime("%Y-%m-%d"), "to": today.strftime("%Y-%m-%d")},
        )
        if not isinstance(news, list):
            return []
        # sort by datetime if present (unix timestamp)
        news = sorted(news, key=lambda x: x.get("datetime", 0), reverse=True)[:10]
        for n in news:
            try:
                n["datetime"] = datetime.datetime.utcfromtimestamp(n.get("datetime", 0)).strftime("%Y-%m-%d %H:%M")
            except Exception:
                n["datetime"] = "未知時間 | Unknown time"
        return news
    except Exception as e:
        logger.warning(f"get_recent_news failed for {symbol}: {e}")
        return []


# --- Helper: safe float rounding ---
def _safe_round(x, ndigits=2):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/A"
        return round(float(x), ndigits)
    except Exception:
        return "N/A"


# --- Main stock fetch and analysis ---
def get_stock_data(symbol: str):
    symbol = symbol.strip()
    if not symbol.isdigit() or len(symbol) != 4:
        return {"error": "股票代號必須為4位數字 | Stock ID must be a 4-digit number"}

    for attempt in range(3):
        try:
            yf_ticker = yf.Ticker(f"{symbol}.TW")
            # use 3 months daily bars
            history = yf_ticker.history(period="3mo", interval="1d", auto_adjust=False)
            info = {}
            try:
                info = yf_ticker.info or {}
            except Exception:
                # some yfinance versions raise on .info
                info = {}

            if history is None or history.empty:
                logger.warning(f"No history for {symbol}.TW (attempt {attempt+1})")
                time.sleep(1)
                continue

            # Ensure index is sorted and recent
            history = history.sort_index()
            close = history["Close"]
            if close.empty or len(close) < 2:
                logger.warning(f"Insufficient price history for {symbol}.TW")
                time.sleep(1)
                continue

            current_price = _safe_round(close.iloc[-1])
            prev_close = _safe_round(close.iloc[-2])
            daily_change = "N/A"
            try:
                daily_change = _safe_round((float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100, 2)
            except Exception:
                daily_change = "N/A"

            quote = {
                "current_price": current_price,
                "daily_change": daily_change,
                "volume": int(history["Volume"].iloc[-1]) if "Volume" in history.columns and not history["Volume"].empty else "N/A",
                "open_price": _safe_round(history["Open"].iloc[-1]) if "Open" in history.columns else "N/A",
                "high_price": _safe_round(history["High"].iloc[-1]) if "High" in history.columns else "N/A",
                "low_price": _safe_round(history["Low"].iloc[-1]) if "Low" in history.columns else "N/A",
                "prev_close": prev_close,
            }

            # Technical indicators (use last available)
            try:
                sma50 = SMAIndicator(close, window=50).sma_indicator()
                rsi = RSIIndicator(close, window=14).rsi()
                macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
                macd_value = macd.macd()
                technical = {
                    "ma50": _safe_round(sma50.iloc[-1]) if len(sma50) >= 1 else "N/A",
                    "rsi": _safe_round(rsi.iloc[-1]) if len(rsi) >= 1 else "N/A",
                    "macd": _safe_round(macd_value.iloc[-1]) if len(macd_value) >= 1 else "N/A",
                    "support": _safe_round(history["Low"].iloc[-50:].min()) if "Low" in history.columns else "N/A",
                    "resistance": _safe_round(history["High"].iloc[-50:].max()) if "High" in history.columns else "N/A",
                }
            except Exception as e:
                logger.warning(f"Technical indicators failed for {symbol}: {e}")
                technical = {"ma50": "N/A", "rsi": "N/A", "macd": "N/A", "support": "N/A", "resistance": "N/A"}

            # Financial metrics (best-effort)
            metrics = {
                "pe": _safe_round(info.get("trailingPE")) if info.get("trailingPE") is not None else "N/A",
                "pb": _safe_round(info.get("priceToBook")) if info.get("priceToBook") is not None else "N/A",
                "revenue_growth": _safe_round(info.get("revenueGrowth")) if info.get("revenueGrowth") is not None else "N/A",
                "eps_growth": _safe_round(info.get("earningsGrowth")) if info.get("earningsGrowth") is not None else "N/A",
            }

            # GPT / model-based analysis (lazy init)
            analysis = _analyze_with_model_or_fallback(symbol, quote, technical, metrics)

            # news
            news = get_recent_news(symbol)

            return {
                "symbol": symbol,
                "quote": quote,
                "technical": technical,
                "metrics": metrics,
                "gpt_analysis": analysis,
                "industry_en": info.get("industry", "Unknown"),
                "news": news,
            }
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}.TW attempt {attempt+1}: {e}")
            time.sleep(1)

    return {"error": f"無法獲取股票 {symbol} 的數據: 多次嘗試失敗 | Failed to fetch data for {symbol}: Multiple attempts failed"}


def _analyze_with_model_or_fallback(symbol, quote, technical, metrics):
    # If model is available, try inference. Otherwise fallback to rule-based simple analysis.
    try:
        # Lazy initialize model on first use
        if llama_pipeline is None and TORCH_AVAILABLE:
            initialize_llama_model()

        # If model not available, fallback
        if llama_pipeline is None:
            return _rule_based_analysis(quote, technical, metrics)

        # Build a simple prompt string (bilingual)
        now_local = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d")
        prompt_lines = [
            "你是一位專業的台灣股市交易分析師。請用中文 及 English 提供 1) 建議 (buy/sell/hold) 2) 理由 3) 風險 4) 簡短總結 (二語)。",
            f"Symbol: {symbol}",
            f"Date: {now_local}",
            f"Open: {quote.get('open_price', 'N/A')}, High: {quote.get('high_price','N/A')}, Low: {quote.get('low_price','N/A')}, Close: {quote.get('current_price','N/A')}, Change%: {quote.get('daily_change','N/A')}",
            f"RSI: {technical.get('rsi','N/A')}, MACD: {technical.get('macd','N/A')}, MA50: {technical.get('ma50','N/A')}",
            f"PE: {metrics.get('pe','N/A')}, PB: {metrics.get('pb','N/A')}",
            "請在每行以 'Recommendation:', 'Rationale:', 'Risk:', 'Summary:' 開頭回覆。 / Please start lines with Recommendation:, Rationale:, Risk:, Summary:."
        ]
        prompt = "\n".join(prompt_lines)

        # Call pipeline - catch variations in return format
        outputs = llama_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.6)
        if not outputs or not isinstance(outputs, list):
            logger.warning("Model returned unexpected outputs, falling back.")
            return _rule_based_analysis(quote, technical, metrics)

        gen = outputs[0]
        generated_text = gen.get("generated_text") or gen.get("text") or ""
        # Try to parse generated_text for labeled lines
        rec = "hold"
        rationale = "模型未提供明確建議。 | Model did not provide a clear recommendation."
        risk = "中等風險，需密切關注市場動態。 | Moderate risk, monitor market dynamics closely."
        summary = "根據模型分析，投資決策應謹慎。 | Based on model analysis, be cautious."

        for line in generated_text.splitlines():
            line = line.strip()
            if line.lower().startswith("recommendation:") or line.startswith("建議:"):
                candidate = line.split(":", 1)[1].strip().lower()
                if "buy" in candidate or "買" in candidate:
                    rec = "buy"
                elif "sell" in candidate or "賣" in candidate:
                    rec = "sell"
                else:
                    rec = "hold"
            elif line.lower().startswith("rationale:") or line.startswith("理由:"):
                rationale = line.split(":", 1)[1].strip()
            elif line.lower().startswith("risk:") or line.startswith("風險:"):
                risk = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:") or line.startswith("總結:"):
                summary = line.split(":", 1)[1].strip()

        return {
            "recommendation": rec,
            "rationale": rationale,
            "risk": risk,
            "summary": summary,
            "raw_model_output": generated_text[:1000],
        }

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return _rule_based_analysis(quote, technical, metrics)


def _rule_based_analysis(quote, technical, metrics):
    # Simple fallback logic (bilingual texts)
    rec = "hold"
    rationale = "由於目前缺乏關鍵的財務和技術指標數據，建議持有觀望。 | Due to lack of complete data, hold."
    risk = "中等風險，請密切關注。 | Moderate risk, monitor closely."
    summary = "依技術指標與基本面初步建議持有。 | Preliminary hold based on indicators."

    try:
        rsi = technical.get("rsi")
        macd = technical.get("macd")
        # rsi/macd may be "N/A", so ensure numeric
        if isinstance(rsi, (int, float)) and isinstance(macd, (int, float)):
            if rsi > 70 and macd < 0:
                rec = "sell"
                rationale = "RSI 顯示超買且 MACD 呈看跌，可能需要獲利了結。 | RSI overbought and MACD bearish."
            elif rsi < 30 and macd > 0:
                rec = "buy"
                rationale = "RSI 顯示超賣且 MACD 呈看漲，可能有反彈機會。 | RSI oversold and MACD bullish."
            else:
                rec = "hold"
                rationale = "技術指標中性，建議觀望。 | Neutral technicals, suggest hold."
    except Exception:
        pass

    return {"recommendation": rec, "rationale": rationale, "risk": risk, "summary": summary}


# --- Flask routes ---
@app.before_request
def ensure_session_defaults():
    # initialize simple session defaults
    session.setdefault("paid_tier", 0)
    session.setdefault("request_count", 0)
    session.setdefault("subscribed", False)


@app.route("/", methods=["GET", "POST"])
def index():
    symbol_input = ""
    result = {}
    ip = get_client_ip()
    current_tier_index = session.get("paid_tier", 0)
    current_tier = PRICING_TIERS[current_tier_index]
    current_tier_name = current_tier["name"]
    current_limit = current_tier["limit"]
    request_count = session.get("request_count", user_data.get(f"user:{ip}:request_count", 0))

    if request.method == "POST":
        symbol = request.form.get("symbol", "").strip()
        symbol_input = symbol
        if not symbol:
            result = {"error": "請輸入股票代號 | Please enter a stock symbol"}
        elif request_count >= current_limit:
            result = {"error": f"已達到 {current_tier_name} 方案的請求限制 ({current_limit}) | Request limit reached for {current_tier_name} tier ({current_limit})"}
        else:
            # Extra per-IP windowed check (prevents abuse across sessions)
            if not check_rate_limit(ip, current_limit):
                result = {"error": "已達到每小時請求限制，請稍後再試 | Hourly request limit reached, try again later."}
            else:
                result = get_stock_data(symbol)
                if "error" not in result:
                    # increment count
                    session["request_count"] = request_count + 1
                    user_data[f"user:{ip}:request_count"] = session["request_count"]

    return render_template(
        "index.html",
        symbol_input=symbol_input,
        result=result,
        current_tier_name=current_tier_name,
        request_count=session.get("request_count", 0),
        current_limit=current_limit,
        QUOTE_FIELDS=QUOTE_FIELDS,
        METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
        tiers=PRICING_TIERS,
        stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
        stripe_mode=STRIPE_MODE,
    )


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not STRIPE_INSTALLED:
        return jsonify({"error": "Stripe integration not available on this server."}), 500

    tier_name = request.form.get("tier")
    tier = next((t for t in PRICING_TIERS if t["name"] == tier_name), None)
    if not tier:
        logger.error(f"Invalid tier requested: {tier_name}")
        return jsonify({"error": "無效的方案 | Invalid tier"}), 400

    if tier_name == "Free":
        session["subscribed"] = False
        session["paid_tier"] = 0
        session["request_count"] = 0
        user_data[f"user:{get_client_ip()}:tier"] = "Free"
        user_data[f"user:{get_client_ip()}:request_count"] = 0
        flash("✅ 已切換到免費方案 | Switched to Free tier.", "success")
        return jsonify({"url": url_for("index", _external=True)})

    price_id = STRIPE_PRICE_IDS.get(tier_name)
    if not price_id:
        logger.error(f"No valid Price ID configured for {tier_name}")
        flash(f"⚠️ {tier_name} 方案目前不可用 | Subscription for {tier_name} is currently unavailable.", "warning")
        return jsonify({"error": f"{tier_name} 方案目前不可用 | Subscription for {tier_name} is currently unavailable"}), 400
    try:
        logger.info(f"Creating Stripe checkout session for {tier_name}")
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=url_for("payment_success", tier_name=tier_name, _external=True),
            cancel_url=url_for("index", _external=True),
            metadata={"tier": tier_name, "user_id": get_client_ip()},
        )
        return jsonify({"url": checkout_session.url})
    except Exception as e:
        logger.error(f"Unexpected Stripe error: {e}")
        return jsonify({"error": f"無法創建結帳會話: {str(e)} | Failed to create checkout session: {str(e)}"}), 500


@app.route("/payment-success/<tier_name>")
def payment_success(tier_name):
    tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t["name"] == tier_name), None)
    if tier_index is not None and tier_name != "Free":
        session["subscribed"] = True
        session["paid_tier"] = tier_index
        session["request_count"] = 0
        user_data[f"user:{get_client_ip()}:tier"] = tier_name
        user_data[f"user:{get_client_ip()}:request_count"] = 0
        flash(f"✅ 成功訂閱 {tier_name} 方案 | Subscription successful for {tier_name} plan.", "success")
        logger.info(f"Subscription successful for {tier_name} (tier index: {tier_index})")
    return redirect(url_for("index"))


@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    if not STRIPE_INSTALLED:
        return jsonify({"error": "Stripe not installed on server."}), 500

    payload = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature")
    event = None
    try:
        if ENDPOINT_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, ENDPOINT_SECRET)
        else:
            # If you haven't set endpoint secret, try to parse payload (less secure)
            event = stripe.Event.construct_from(request.get_json(force=True), stripe.api_key)
    except Exception as e:
        logger.error(f"Webhook signature/parse error: {e}")
        return jsonify({"error": str(e)}), 400

    try:
        # handle only checkout.session.completed
        if event and event["type"] == "checkout.session.completed":
            stripe_session = event["data"]["object"]
            user_id = stripe_session.get("metadata", {}).get("user_id", "unknown")
            tier = stripe_session.get("metadata", {}).get("tier", "Free")
            tier_index = next((i for i, t in enumerate(PRICING_TIERS) if t["name"] == tier), None)
            if tier_index is not None:
                user_data[f"user:{user_id}:tier"] = tier
                user_data[f"user:{user_id}:request_count"] = 0
                logger.info(f"Webhook: Updated {user_id} to {tier} tier")
    except Exception as e:
        logger.error(f"Error processing webhook event: {e}")
        return jsonify({"error": str(e)}), 400

    return jsonify({"status": "success"})


@app.route("/reset", methods=["POST"])
def reset():
    password = request.form.get("password")
    if password and password == os.getenv("RESET_PASSWORD"):
        ip = get_client_ip()
        session["request_count"] = 0
        session["subscribed"] = False
        session["paid_tier"] = 0
        user_data[f"user:{ip}:request_count"] = 0
        user_data[f"user:{ip}:tier"] = "Free"
        flash("✅ 請求次數已重置 | Request count reset.", "success")
        logger.info(f"Reset request count for user {ip}")
    else:
        flash("❌ 密碼錯誤 | Incorrect password.", "danger")
        logger.warning("Failed reset attempt with incorrect password")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Port and host from environment
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    # Optionally warm the model in production by setting WARM_LLAMA=true
    if os.getenv("WARM_LLAMA", "false").lower() == "true":
        try:
            initialize_llama_model()
        except Exception as e:
            logger.warning(f"Warm initialization failed: {e}")
    app.run(host=host, port=port)
