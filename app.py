import os
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, redirect
import stripe
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

# In-memory storage for user tiers and request counts
user_data = {}

# Stripe setup
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
ENDPOINT_SECRET = os.getenv('STRIPE_ENDPOINT_SECRET')

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

TIERS = [
    {'name': 'Free', 'price': 0, 'limit': 50},
    {'name': 'Basic', 'price': 5, 'limit': 200},
    {'name': 'Pro', 'price': 10, 'limit': 500},
    {'name': 'Premium', 'price': 20, 'limit': 1000},
    {'name': 'Enterprise', 'price': 50, 'limit': 3000}
]

# Initialize Llama 3 8B model with QLoRA
def initialize_llama_model():
    try:
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        use_4bit = True
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        based_model_path = "DavidLanz/Llama3-tw-8B-Instruct"
        adapter_path = "DavidLanz/llama3_8b_taiwan_stock_qlora"

        base_model = AutoModelForCausalLM.from_pretrained(
            based_model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(based_model_path, trust_remote_code=True)

        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer=tokenizer,
        )

        return text_gen_pipeline, tokenizer
    except Exception as e:
        print(f"Failed to initialize Llama model: {e}")
        return None, None

llama_pipeline, llama_tokenizer = initialize_llama_model()

def get_stock_data(symbol):
    # Validate stock ID: must be a 4-digit number
    if not symbol.isdigit() or len(symbol) != 4:
        return {'error': '股票代號必須為4位數字 | Stock ID must be a 4-digit number'}

    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        info = ticker.info
        history = ticker.history(period="3mo")

        if history.empty or not info:
            return {'error': f'無法獲取股票 {symbol} 的數據 | No data available for {symbol}'}

        current_price = history['Close'][-1] if not history['Close'].empty else None
        prev_close = history['Close'][-2] if len(history['Close']) >= 2 else None
        daily_change = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else None

        quote = {
            'current_price': round(current_price, 2) if current_price else 'N/A',
            'daily_change': round(daily_change, 2) if daily_change else 'N/A',
            'volume': int(history['Volume'][-1]) if not history['Volume'].empty else 'N/A',
            'open_price': round(history['Open'][-1], 2) if not history['Open'].empty else 'N/A',
            'high_price': round(history['High'][-1], 2) if not history['High'].empty else 'N/A',
            'low_price': round(history['Low'][-1], 2) if not history['Low'].empty else 'N/A',
            'prev_close': round(prev_close, 2) if prev_close else 'N/A'
        }

        sma50 = SMAIndicator(history['Close'], window=50).sma_indicator()
        rsi = RSIIndicator(history['Close'], window=14).rsi()
        macd = MACD(history['Close'], window_slow=26, window_fast=12, window_sign=9)
        macd_value = macd.macd()

        technical = {
            'ma50': round(sma50[-1], 2) if not sma50.empty else 'N/A',
            'rsi': round(rsi[-1], 2) if not rsi.empty else 'N/A',
            'macd': round(macd_value[-1], 2) if not macd_value.empty else 'N/A',
            'support': round(history['Low'][-50:].min(), 2) if not history['Low'].empty else 'N/A',
            'resistance': round(history['High'][-50:].max(), 2) if not history['High'].empty else 'N/A'
        }

        metrics = {
            'pe': round(info.get('trailingPE', 'N/A'), 2) if info.get('trailingPE') else 'N/A',
            'pb': round(info.get('priceToBook', 'N/A'), 2) if info.get('priceToBook') else 'N/A',
            'revenue_growth': round(info.get('revenueGrowth', 'N/A'), 2) if info.get('revenueGrowth') else 'N/A',
            'eps_growth': round(info.get('earningsGrowth', 'N/A'), 2) if info.get('earningsGrowth') else 'N/A'
        }

        def analyze_stock(quote, technical, metrics):
            if llama_pipeline is None or llama_tokenizer is None:
                recommendation = 'hold'
                rationale = '由於目前缺乏關鍵的財務和技術指標數據，無法確定該股票的具體價值和趨勢，因此建議持有觀望。 | Due to the lack of key financial and technical indicator data, it is difficult to determine the specific value and trend of this stock, thus a hold recommendation is advised.'
                risk = '缺乏數據可能導致投資決策的不確定性，投資者需謹慎評估。 | The lack of data may lead to uncertainty in investment decisions, and investors should assess carefully.'
                summary = '在缺乏具體財務和技術指標的情況下，建議投資者持有該股票，並持續關注未來的市場動態。 | In the absence of specific financial and technical indicators, investors are advised to hold this stock and continue to monitor future market developments.'
                if technical['rsi'] != 'N/A' and technical['macd'] != 'N/A':
                    if technical['rsi'] > 70 and technical['macd'] < 0:
                        recommendation = 'sell'
                        rationale = 'RSI 顯示超買且 MACD 呈看跌，建議賣出。 | RSI indicates overbought and MACD is bearish, suggesting a sell.'
                    elif technical['rsi'] < 30 and technical['macd'] > 0:
                        recommendation = 'buy'
                        rationale = 'RSI 顯示超賣且 MACD 呈看漲，建議買入。 | RSI indicates oversold and MACD is bullish, suggesting a buy.'
                    elif technical['rsi'] >= 30 and technical['rsi'] <= 70:
                        recommendation = 'hold'
                        rationale = '市場條件中性，建議持有觀望。 | Market conditions are neutral, suggesting a hold.'
                    risk = '中等風險，需密切關注市場動態。 | Moderate risk, monitor market dynamics closely.'
                    summary = '根據技術指標，投資決策應謹慎，持續關注市場變化。 | Based on technical indicators, investment decisions should be cautious, with ongoing market monitoring.'
                return {
                    'recommendation': recommendation,
                    'rationale': rationale,
                    'risk': risk,
                    'summary': summary
                }

            last_trading_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            messages = [
                {
                    "role": "system",
                    "content": "你是一位專業的台灣股市交易分析師"
                },
                {
                    "role": "user",
                    "content": (
                        f"{symbol} 上一個交易日的表現，"
                        f"開盤價是 {quote['open_price'] if quote['open_price'] != 'N/A' else '未知'}, "
                        f"當日最高價是 {quote['high_price'] if quote['high_price'] != 'N/A' else '未知'}, "
                        f"當日最低價是 {quote['low_price'] if quote['low_price'] != 'N/A' else '未知'}, "
                        f"收盤價是 {quote['current_price'] if quote['current_price'] != 'N/A' else '未知'}, "
                        f"與前一天相比變化 {quote['daily_change'] if quote['daily_change'] != 'N/A' else '未知'}%, "
                        f"成交股數為 {quote['volume'] if quote['volume'] != 'N/A' else '未知'}, "
                        f"RSI: {technical['rsi'] if technical['rsi'] != 'N/A' else '未知'}, "
                        f"MACD: {technical['macd'] if technical['macd'] != 'N/A' else '未知'}, "
                        f"PE: {metrics['pe'] if metrics['pe'] != 'N/A' else '未知'}, "
                        f"PB: {metrics['pb'] if metrics['pb'] != 'N/A' else '未知'}. "
                        f"請預測今天的收盤價並提供投資建議、風險評估和總結。"
                    )
                }
            ]

            prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            terminators = [
                llama_tokenizer.eos_token_id,
                llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            try:
                outputs = llama_pipeline(
                    prompt,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                generated_text = outputs[0]["generated_text"][len(prompt):].strip()

                recommendation = 'hold'
                rationale = '模型未提供明確建議。 | Model did not provide a clear recommendation.'
                risk = '中等風險，需密切關注市場動態。 | Moderate risk, monitor market dynamics closely.'
                summary = '根據模型分析，投資決策應謹慎，持續關注市場變化。 | Based on model analysis, investment decisions should be cautious, with ongoing market monitoring.'
                predicted_price = 'N/A'

                for line in generated_text.split('\n'):
                    line = line.strip()
                    if line.startswith('預測收盤價:') or line.startswith('Predicted Close:'):
                        predicted_price = line.split(':')[-1].strip()
                    elif line.startswith('建議:') or line.startswith('Recommendation:'):
                        recommendation = line.split(':')[-1].strip().lower()
                        if '買入' in recommendation or 'buy' in recommendation:
                            recommendation = 'buy'
                        elif '賣出' in recommendation or 'sell' in recommendation:
                            recommendation = 'sell'
                        else:
                            recommendation = 'hold'
                    elif line.startswith('理由:') or line.startswith('Rationale:'):
                        rationale = line.split(':')[-1].strip()
                    elif line.startswith('風險:') or line.startswith('Risk:'):
                        risk = line.split(':')[-1].strip()
                    elif line.startswith('總結:') or line.startswith('Summary:'):
                        summary = line.split(':')[-1].strip()

                return {
                    'recommendation': recommendation,
                    'rationale': rationale,
                    'risk': risk,
                    'summary': summary,
                    'predicted_price': predicted_price
                }
            except Exception as e:
                print(f"Llama model inference failed: {e}")
                recommendation = 'hold'
                rationale = '模型推斷失敗，採用後備邏輯。 | Model inference failed, using fallback logic.'
                risk = '中等風險，需密切關注市場動態。 | Moderate risk, monitor market dynamics closely.'
                summary = '由於模型推斷失敗，建議謹慎並持續關注市場。 | Due to model inference failure, exercise caution and monitor market trends.'
                if technical['rsi'] != 'N/A' and technical['macd'] != 'N/A':
                    if technical['rsi'] > 70 and technical['macd'] < 0:
                        recommendation = 'sell'
                        rationale = 'RSI 顯示超買且 MACD 呈看跌，建議賣出。 | RSI indicates overbought and MACD is bearish, suggesting a sell.'
                    elif technical['rsi'] < 30 and technical['macd'] > 0:
                        recommendation = 'buy'
                        rationale = 'RSI 顯示超賣且 MACD 呈看漲，建議買入。 | RSI indicates oversold and MACD is bullish, suggesting a buy.'
                    elif technical['rsi'] >= 30 and technical['rsi'] <= 70:
                        recommendation = 'hold'
                        rationale = '市場條件中性，建議持有觀望。 | Market conditions are neutral, suggesting a hold.'
                    risk = '中等風險，需密切關注市場動態。 | Moderate risk, monitor market dynamics closely.'
                    summary = '根據技術指標，投資決策應謹慎，持續關注市場變化。 | Based on technical indicators, investment decisions should be cautious, with ongoing market monitoring.'
                return {
                    'recommendation': recommendation,
                    'rationale': rationale,
                    'risk': risk,
                    'summary': summary
                }

        gpt_analysis = analyze_stock(quote, technical, metrics)

        return {
            'symbol': symbol,
            'quote': quote,
            'technical': technical,
            'metrics': metrics,
            'gpt_analysis': gpt_analysis,
            'industry_en': info.get('industry', 'Unknown'),
            'news': []  # yfinance does not provide news; use Finnhub or other API for news
        }

    except Exception as e:
        return {'error': f'無法獲取股票 {symbol} 的數據: {str(e)} | Failed to fetch data for {symbol}: {str(e)}'}

@app.route('/', methods=['GET', 'POST'])
def index():
    symbol_input = ''
    result = {}
    user_id = request.remote_addr
    current_tier = user_data.get(f'user:{user_id}:tier', 'Free')
    request_count = user_data.get(f'user:{user_id}:request_count', 0)
    current_limit = next(tier['limit'] for tier in TIERS if tier['name'] == current_tier)

    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip()
        symbol_input = symbol
        if not symbol:
            result = {'error': '請輸入股票代號 | Please enter a stock symbol'}
        elif request_count >= current_limit:
            result = {'error': f'已達到 {current_tier} 方案的請求限制 ({current_limit}) | Request limit reached for {current_tier} tier ({current_limit})'}
        else:
            result = get_stock_data(symbol)
            if 'error' not in result:
                user_data[f'user:{user_id}:request_count'] = request_count + 1

    return render_template(
        "index.html",
        symbol_input=symbol_input,
        result=result,
        current_tier_name=current_tier,
        request_count=request_count,
        current_limit=current_limit,
        QUOTE_FIELDS=QUOTE_FIELDS,
        METRIC_NAMES_ZH_EN=METRIC_NAMES_ZH_EN,
        tiers=TIERS
    )

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    tier = request.form.get('tier')
    selected_tier = next((t for t in TIERS if t['name'] == tier), None)
    if not selected_tier or selected_tier['price'] == 0:
        return jsonify({'error': '無效的方案 | Invalid tier'})
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': f'{tier} Tier'},
                    'unit_amount': selected_tier['price'] * 100,
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://your-app.railway.app/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='https://your-app.railway.app/',
            metadata={'tier': tier, 'user_id': request.remote_addr}
        )
        return jsonify({'url': checkout_session.url})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/success')
def success():
    session_id = request.args.get('session_id')
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        user_id = session.metadata.user_id
        tier = session.metadata.tier
        user_data[f'user:{user_id}:tier'] = tier
        user_data[f'user:{user_id}:request_count'] = 0
    except Exception as e:
        print(f"Error in success route: {e}")
    return redirect('/')

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, ENDPOINT_SECRET)
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            user_id = session['metadata']['user_id']
            tier = session['metadata']['tier']
            user_data[f'user:{user_id}:tier'] = tier
            user_data[f'user:{user_id}:request_count'] = 0
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 400
    return jsonify({'status': 'success'})

@app.route('/reset', methods=['POST'])
def reset():
    password = request.form.get('password')
    if password == os.getenv('RESET_PASSWORD'):
        user_id = request.remote_addr
        user_data[f'user:{user_id}:request_count'] = 0
    return redirect('/')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv('PORT', 8080)))
