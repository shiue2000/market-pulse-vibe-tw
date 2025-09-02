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
import pandas as pd
import json, os
import urllib.parse
from collections import namedtuple
from itertools import cycle
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

# Proxy module
class ProxyProvider:
    def get_proxy(self):
        return NotImplemented

class NoProxyProvider(ProxyProvider):
    def get_proxy(self):
        return {}

class SingleProxyProvider(ProxyProvider):
    def __init__(self, proxy=None):
        self._proxy = proxy

    def get_proxy(self):
        return self._proxy

class RoundRobinProxiesProvider(ProxyProvider):
    def __init__(self, proxies: list):
        self._proxies = proxies
        self._proxies_cycle = cycle(proxies)

    @property
    def proxies(self):
        return self._proxies

    @proxies.setter
    def proxies(self, proxies: list):
        if not isinstance(proxies, list):
            raise ValueError("Proxies only accept list")
        self._proxies = proxies
        self._proxies_cycle = cycle(proxies)

    def get_proxy(self):
        return next(self._proxies_cycle)

_provider_instance = NoProxyProvider()

def reset_proxy_provider():
    configure_proxy_provider(NoProxyProvider())

def configure_proxy_provider(provider_instance):
    global _provider_instance
    if not isinstance(provider_instance, ProxyProvider):
        raise BaseException("proxy provider should be a ProxyProvider object")
    _provider_instance = provider_instance

def get_proxies():
    return _provider_instance.get_proxy()

# Analytics module
class Analytics(object):
    def continuous(self, data):
        diff = [1 if data[-i] > data[-i - 1] else -1 for i in range(1, len(data))]
        cont = 0
        for v in diff:
            if v == diff[0]:
                cont += 1
            else:
                break
        return cont * diff[0]

    def moving_average(self, data, days):
        result = []
        data = data[:]
        for _ in range(len(data) - days + 1):
            result.append(round(sum(data[-days:]) / days, 2))
            data.pop()
        return result[::-1]

    def ma_bias_ratio(self, day1, day2):
        data1 = self.moving_average(self.price, day1)
        data2 = self.moving_average(self.price, day2)
        result = [data1[-i] - data2[-i] for i in range(1, min(len(data1), len(data2)) + 1)]
        return result[::-1]

    def ma_bias_ratio_pivot(self, data, sample_size=5, position=False):
        sample = data[-sample_size:]
        if position is True:
            check_value = max(sample)
            pre_check_value = max(sample) > 0
        elif position is False:
            check_value = min(sample)
            pre_check_value = max(sample) < 0
        return (
            (
                sample_size - sample.index(check_value) < 4
                and sample.index(check_value) != sample_size - 1
                and pre_check_value
            ),
            sample_size - sample.index(check_value) - 1,
            check_value,
        )

class BestFourPoint(object):
    BEST_BUY_WHY = ["量大收紅", "量縮價不跌", "三日均價由下往上", "三日均價大於六日均價"]
    BEST_SELL_WHY = ["量大收黑", "量縮價跌", "三日均價由上往下", "三日均價小於六日均價"]

    def __init__(self, stock):
        self.stock = stock

    def bias_ratio(self, position=False):
        return self.stock.ma_bias_ratio_pivot(self.stock.ma_bias_ratio(3, 6), position=position)

    def plus_bias_ratio(self):
        return self.bias_ratio(True)

    def mins_bias_ratio(self):
        return self.bias_ratio(False)

    def best_buy_1(self):
        return (
            self.stock.capacity[-1] > self.stock.capacity[-2]
            and self.stock.price[-1] > self.stock.open[-1]
        )

    def best_buy_2(self):
        return (
            self.stock.capacity[-1] < self.stock.capacity[-2]
            and self.stock.price[-1] > self.stock.open[-2]
        )

    def best_buy_3(self):
        return self.stock.continuous(self.stock.moving_average(self.stock.price, 3)) == 1

    def best_buy_4(self):
        return (
            self.stock.moving_average(self.stock.price, 3)[-1]
            > self.stock.moving_average(self.stock.price, 6)[-1]
        )

    def best_sell_1(self):
        return (
            self.stock.capacity[-1] > self.stock.capacity[-2]
            and self.stock.price[-1] < self.stock.open[-1]
        )

    def best_sell_2(self):
        return (
            self.stock.capacity[-1] < self.stock.capacity[-2]
            and self.stock.price[-1] < self.stock.open[-2]
        )

    def best_sell_3(self):
        return self.stock.continuous(self.stock.moving_average(self.stock.price, 3)) == -1

    def best_sell_4(self):
        return (
            self.stock.moving_average(self.stock.price, 3)[-1]
            < self.stock.moving_average(self.stock.price, 6)[-1]
        )

    def best_four_point_to_buy(self):
        result = []
        check = [
            self.best_buy_1(),
            self.best_buy_2(),
            self.best_buy_3(),
            self.best_buy_4(),
        ]
        if self.mins_bias_ratio() and any(check):
            for index, v in enumerate(check):
                if v:
                    result.append(self.BEST_BUY_WHY[index])
        else:
            return False
        return ", ".join(result)

    def best_four_point_to_sell(self):
        result = []
        check = [
            self.best_sell_1(),
            self.best_sell_2(),
            self.best_sell_3(),
            self.best_sell_4(),
        ]
        if self.plus_bias_ratio() and any(check):
            for index, v in enumerate(check):
                if v:
                    result.append(self.BEST_SELL_WHY[index])
        else:
            return False
        return ", ".join(result)

    def best_four_point(self):
        buy = self.best_four_point_to_buy()
        sell = self.best_four_point_to_sell()
        if buy:
            return (True, buy)
        elif sell:
            return (False, sell)
        return None

# Stock module
TWSE_BASE_URL = "http://www.twse.com.tw/"
TPEX_BASE_URL = "http://www.tpex.org.tw/"
DATATUPLE = namedtuple(
    "Data",
    ["date", "capacity", "turnover", "open", "high", "low", "close", "change", "transaction"],
)

class BaseFetcher(object):
    def fetch(self, year, month, sid, retry):
        pass

    def _convert_date(self, date):
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
            try:
                r = requests.get(self.REPORT_URL, params=params, proxies=get_proxies(), timeout=10)
                r.raise_for_status()
                data = r.json()
                logger.debug(f"[TWSE Fetch] Response for {sid}: {data}")
                if data.get("stat") == "OK":
                    data["data"] = self.purify(data)
                    return data
                logger.warning(f"[TWSE Fetch] Non-OK stat for {sid}: {data.get('stat')}")
            except (JSONDecodeError, requests.RequestException) as e:
                logger.warning(f"[TWSE Fetch Error] Attempt {retry_i + 1} for {sid}: {e}")
                time.sleep(3)
        logger.error(f"[TWSE Fetch Error] Failed to fetch data for {sid} after {retry} attempts")
        return {"stat": "", "data": []}

    def _make_datatuple(self, data):
        try:
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
        except Exception as e:
            logger.error(f"[TWSE Parse Error] Failed to parse data: {e}")
            return None

    def purify(self, original_data):
        return [d for d in (self._make_datatuple(d) for d in original_data["data"]) if d is not None]

class TPEXFetcher(BaseFetcher):
    REPORT_URL = urllib.parse.urljoin(TPEX_BASE_URL, "web/stock/aftertrading/daily_trading_info/st43_result.php")

    def fetch(self, year: int, month: int, sid: str, retry: int = 5):
        params = {"d": "%d/%d" % (year - 1911, month), "stkno": sid}
        for retry_i in range(retry):
            try:
                r = requests.get(self.REPORT_URL, params=params, proxies=get_proxies(), timeout=10)
                r.raise_for_status()
                data = r.json()
                logger.debug(f"[TPEX Fetch] Response for {sid}: {data}")
                data["data"] = self.purify(data) if data.get("aaData") else []
                return data
            except (JSONDecodeError, requests.RequestException) as e:
                logger.warning(f"[TPEX Fetch Error] Attempt {retry_i + 1} for {sid}: {e}")
                time.sleep(3)
        logger.error(f"[TPEX Fetch Error] Failed to fetch data for {sid} after {retry} attempts")
        return {"aaData": [], "data": []}

    def _convert_date(self, date):
        return "/".join([str(int(date.split("/")[0]) + 1911)] + date.split("/")[1:])

    def _make_datatuple(self, data):
        try:
            data[0] = datetime.datetime.strptime(self._convert_date(data[0].replace("＊", "")), "%Y/%m/%d")
            data[1] = int(data[1].replace(",", "")) * 1000
            data[2] = int(data[2].replace(",", "")) * 1000
            data[3] = None if data[3] == "--" else float(data[3].replace(",", ""))
            data[4] = None if data[4] == "--" else float(data[4].replace(",", ""))
            data[5] = None if data[5] == "--" else float(data[5].replace(",", ""))
            data[6] = None if data[6] == "--" else float(data[6].replace(",", ""))
            data[7] = float(data[7].replace(",", ""))
            data[8] = int(data[8].replace(",", ""))
            return DATATUPLE(*data)
        except Exception as e:
            logger.error(f"[TPEX Parse Error] Failed to parse data: {e}")
            return None

    def purify(self, original_data):
        return [d for d in (self._make_datatuple(d) for d in original_data["aaData"]) if d is not None]

class Stock(Analytics):
    def __init__(self, sid: str, initial_fetch: bool = True):
        self.sid = sid
        self.fetcher = None
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
        self.raw_data = []
        self.data = []
        fetchers = [TWSEFetcher(), TPEXFetcher()]
        for fetcher in fetchers:
            self.fetcher = fetcher
            fetched_data = self.fetcher.fetch(year, month, self.sid, retry=5)
            if fetched_data.get("data"):
                self.raw_data = [fetched_data]
                self.data = fetched_data["data"]
                logger.info(f"Successfully fetched data for {self.sid} using {fetcher.__class__.__name__}")
                break
        else:
            logger.error(f"No data fetched for {self.sid} from either TWSE or TPEX")
        return self.data

    def fetch_from(self, year: int, month: int):
        self.raw_data = []
        self.data = []
        today = datetime.datetime.today()
        for y, m in self._month_year_iter(month, year, today.month, today.year):
            self.fetch(y, m)
            self.raw_data.append(self.raw_data[-1] if self.raw_data else {"data": []})
            self.data.extend(self.data)
        return self.data

    def fetch_31(self):
        today = datetime.datetime.today()
        before = today - datetime.timedelta(days=60)
        self.fetch_from(before.year, before.month)
        self.data = self.data[-31:]
        return self.data

    @property
    def date(self):
        return [d.date for d in self.data if d.date is not None]

    @property
    def capacity(self):
        return [d.capacity for d in self.data if d.capacity is not None]

    @property
    def turnover(self):
        return [d.turnover for d in self.data if d.turnover is not None]

    @property
    def price(self):
        return [d.close for d in self.data if d.close is not None]

    @property
    def high(self):
        return [d.high for d in self.data if d.high is not None]

    @property
    def low(self):
        return [d.low for d in self.data if d.low is not None]

    @property
    def open(self):
        return [d.open for d in self.data if d.open is not None]

    @property
    def close(self):
        return [d.close for d in self.data if d.close is not None]

    @property
    def change(self):
        return [d.change for d in self.data if d.change is not None]

    @property
    def transaction(self):
        return [d.transaction for d in self.data if d.transaction is not None]

# Realtime module
SESSION_URL = "http://mis.twse.com.tw/stock/index.jsp"
STOCKINFO_URL = "http://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={stock_id}&_={time}"

def _format_stock_info(data):
    logger.debug(f"Formatting stock info: {data}")
    result = {"timestamp": 0.0, "info": {}, "realtime": {}}
    try:
        result["timestamp"] = int(data["tlong"]) / 1000
    except:
        result["timestamp"] = 0.0
    result["info"]["code"] = data.get("c", "")
    result["info"]["channel"] = data.get("ch", "")
    result["info"]["name"] = data.get("n", "")
    result["info"]["fullname"] = data.get("nf", "")
    result["info"]["time"] = (
        datetime.datetime.fromtimestamp(result["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        if result["timestamp"]
        else ""
    )
    def _split_best(d):
        return d.strip("_").split("_") if d else []
    rt = result["realtime"]
    rt["latest_trade_price"] = data.get("z", "--")
    rt["trade_volume"] = data.get("tv", "--")
    rt["accumulate_trade_volume"] = data.get("v", "--")
    rt["best_bid_price"] = _split_best(data.get("b", ""))
    rt["best_bid_volume"] = _split_best(data.get("g", ""))
    rt["best_ask_price"] = _split_best(data.get("a", ""))
    rt["best_ask_volume"] = _split_best(data.get("f", ""))
    rt["open"] = data.get("o", "--")
    rt["high"] = data.get("h", "--")
    rt["low"] = data.get("l", "--")
    rt["previous_close"] = data.get("y", "--")
    result["success"] = True
    return result

def _join_stock_id(stocks, prefix='tse'):
    if isinstance(stocks, list):
        return "|".join([f"{prefix}_{s}.tw" for s in stocks])
    return f"{prefix}_{stocks}.tw"

def get_raw(stocks):
    req = requests.Session()
    try:
        r = req.get(SESSION_URL, proxies=get_proxies(), timeout=15)
        r.raise_for_status()
        logger.info(f"[TWSE Session] Successfully initialized session for {stocks}")
    except requests.RequestException as e:
        logger.error(f"[TWSE Session Error] Failed to initialize session for {stocks}: {e}")
        return {"rtmessage": "Session initialization failed", "rtcode": "5002"}
    data = {"rtmessage": "Empty Query.", "rtcode": "5001"}
    for prefix in ['tse', 'otc']:
        stock_id = _join_stock_id(stocks, prefix)
        try:
            r = req.get(STOCKINFO_URL.format(stock_id=stock_id, time=int(time.time()) * 1000), timeout=15)
            r.raise_for_status()
            data = r.json()
            logger.debug(f"[TWSE Realtime] Response for {prefix}_{stocks}.tw: {data}")
            if "msgArray" in data and len(data["msgArray"]) > 0:
                break
        except (JSONDecodeError, requests.RequestException) as e:
            logger.warning(f"[TWSE Realtime Error] Failed for {prefix}_{stocks}.tw: {e}")
            time.sleep(5)
    return data

def get(stocks, retry=3):
    for attempt in range(retry):
        data = get_raw(stocks)
        logger.debug(f"[Realtime Attempt {attempt + 1}] Raw data for {stocks}: {data}")
        if data.get("rtcode") == "5000" and "msgArray" in data and len(data["msgArray"]) > 0:
            if isinstance(stocks, list):
                result = {d["c"]: _format_stock_info(d) for d in data["msgArray"]}
                result["success"] = True
                return result
            formatted = _format_stock_info(data["msgArray"][0])
            formatted["success"] = True
            return formatted
        logger.warning(f"[Realtime Retry] Attempt {attempt + 1} failed for {stocks}: {data.get('rtmessage', 'No message')}")
        time.sleep(5)
    data["success"] = False
    data["rtmessage"] = data.get("rtmessage", "Failed to fetch data after retries")
    return data

# Legacy module (optional, if needed for compatibility)
class LegacyAnalytics(object):
    def cal_continue(self, list_data):
        diff_data = []
        for i in range(1, len(list_data)):
            if list_data[-i] > list_data[-i - 1]:
                diff_data.append(1)
            else:
                diff_data.append(-1)
        cont = 0
        for value in diff_data:
            if value == diff_data[0]:
                cont += 1
            else:
                break
        return cont * diff_data[0]

    def moving_average(self, data, days):
        result = []
        data = data[:]
        for dummy in range(len(data) - int(days) + 1):
            result.append(round(sum(data[-days:]) / days, 2))
            data.pop()
        result.reverse()
        return result

    def ma_bias_ratio(self, date1, date2, data):
        data1 = self.moving_average(data, date1)
        data2 = self.moving_average(data, date2)
        cal_list = []
        for i in range(1, min(len(data1), len(data2)) + 1):
            cal_list.append(data1[-i] - data2[-i])
        cal_list.reverse()
        return cal_list

    def ma_bias_ratio_point(self, data, sample=5, positive_or_negative=False):
        sample_data = data[-sample:]
        if positive_or_negative:  # 正
            ckvalue = max(sample_data)  # 尋找最大值
            preckvalue = max(sample_data) > 0  # 區間最大值必須為正
        else:
            ckvalue = min(sample_data)  # 尋找最小值
            preckvalue = max(sample_data) < 0  # 區間最大值必須為負
        return (
            sample - sample_data.index(ckvalue) < 4
            and sample_data.index(ckvalue) != sample - 1
            and preckvalue,
            sample - sample_data.index(ckvalue) - 1,
            ckvalue,
        )

class LegacyBestFourPoint(object):
    def __init__(self, data):
        self.data = data

    def bias_ratio(self, position=False):
        return self.data.ma_bias_ratio_point(
            self.data.ma_bias_ratio(3, 6, self.data.price), position=position
        )

    def check_plus_bias_ratio(self):
        return self.bias_ratio(True)

    def check_mins_bias_ratio(self):
        return self.bias_ratio()

    def best_buy_1(self):
        result = (
            self.data.capacity[-1] > self.data.capacity[-2]
            and self.data.price[-1] > self.data.open[-1]
        )
        return result

    def best_buy_2(self):
        result = (
            self.data.capacity[-1] < self.data.capacity[-2]
            and self.data.price[-1] > self.data.price[-2]
        )
        return result

    def best_buy_3(self):
        return self.data.continuous(self.data.moving_average(self.data.price, 3)) == 1

    def best_buy_4(self):
        return (
            self.data.moving_average(self.data.price, 3)[-1]
            > self.data.moving_average(self.data.price, 6)[-1]
        )

    def best_sell_1(self):
        result = (
            self.data.capacity[-1] > self.data.capacity[-2]
            and self.data.price[-1] < self.data.open[-1]
        )
        return result

    def best_sell_2(self):
        result = (
            self.data.capacity[-1] < self.data.capacity[-2]
            and self.data.price[-1] < self.data.price[-2]
        )
        return result

    def best_sell_3(self):
        return self.data.continuous(self.data.moving_average(self.data.price, 3)) == -1

    def best_sell_4(self):
        return (
            self.data.moving_average(self.data.price, 3)[-1]
            < self.data.moving_average(self.data.price, 6)[-1]
        )

    def best_four_point_to_buy(self):
        result = []
        if self.check_mins_bias_ratio() and (
            self.best_buy_1()
            or self.best_buy_2()
            or self.best_buy_3()
            or self.best_buy_4()
        ):
            if self.best_buy_1():
                result.append(self.best_buy_1.__doc__.strip())
            if self.best_buy_2():
                result.append(self.best_buy_2.__doc__.strip())
            if self.best_buy_3():
                result.append(self.best_buy_3.__doc__.strip())
            if self.best_buy_4():
                result.append(self.best_buy_4.__doc__.strip())
            result = ", ".join(result)
        else:
            result = False
        return result

    def best_four_point_to_sell(self):
        result = []
        if self.check_plus_bias_ratio() and (
            self.best_sell_1()
            or self.best_sell_2()
            or self.best_sell_3()
            or self.best_sell_4()
        ):
            if self.best_sell_1():
                result.append(self.best_sell_1.__doc__.strip())
            if self.best_sell_2():
                result.append(self.best_sell_2.__doc__.strip())
            if self.best_sell_3():
                result.append(self.best_sell_3.__doc__.strip())
            if self.best_sell_4():
                result.append(self.best_sell_4.__doc__.strip())
            result = ", ".join(result)
        else:
            result = False
        return result

    def best_four_point(self):
        buy = self.best_four_point_to_buy()
        sell = self.best_four_point_to_sell()

        if buy:
            return True, buy
        elif sell:
            return False, sell

        return None

# Flask app
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
STRIPE_TEST_SECRET_KEY = os.getenv("STRIPE_TEST_SECRET_KEY")
STRIPE_TEST_PUBLISHABLE_KEY = os.getenv("STRIPE_TEST_PUBLISHABLE_KEY")
STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY")
STRIPE_LIVE_PUBLISHABLE_KEY = os.getenv("STRIPE_LIVE_PUBLISHABLE_KEY")
STRIPE_MODE = os.getenv("STRIPE_MODE", "test").lower()
STRIPE_PRICE_IDS = {
    "Free": os.getenv("STRIPE_PRICE_TIER0"),
    "Tier 1": os.getenv("STRIPE_PRICE_TIER1"),
    "Tier 2": os.getenv("STRIPE_PRICE_TIER2"),
    "Tier 3": os.getenv("STRIPE_PRICE_TIER3"),
    "Tier 4": os.getenv("STRIPE_PRICE_TIER4"),
}
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not set in .env")
if STRIPE_MODE == "live":
    STRIPE_SECRET_KEY = STRIPE_LIVE_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_LIVE_PUBLISHABLE_KEY
else:
    STRIPE_SECRET_KEY = STRIPE_TEST_SECRET_KEY
    STRIPE_PUBLISHABLE_KEY = STRIPE_TEST_PUBLISHABLE_KEY
if not STRIPE_SECRET_KEY or not STRIPE_PUBLISHABLE_KEY:
    raise RuntimeError(f"❌ Stripe keys for mode '{STRIPE_MODE}' not set in .env")
stripe.api_key = STRIPE_SECRET_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY
openai.api_key = OPENAI_API_KEY

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
PRICING_TIERS = [
    {"name": "Free", "limit": 50, "price": 0},
    {"name": "Tier 1", "limit": 100, "price": 9.99},
    {"name": "Tier 2", "limit": 200, "price": 19.99},
    {"name": "Tier 3", "limit": 400, "price": 29.99},
    {"name": "Tier 4", "limit": 800, "price": 39.99},
]

def validate_price_id(price_id, tier_name):
    return bool(price_id)

def get_finnhub_json(endpoint, params):
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params["token"] = os.getenv("FINNHUB_API_KEY")
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            logger.debug(f"[Finnhub] Response for {endpoint} with {params}: {data}")
            return data
        except Exception as e:
            logger.warning(f"[Finnhub Error] Attempt {attempt + 1} for {endpoint}: {e}")
            time.sleep(3)
    logger.error(f"[Finnhub Error] Failed to fetch {endpoint} after 3 attempts")
    return {}

def get_quote(symbol):
    logger.debug(f"[get_quote] Fetching quote for {symbol}")
    try:
        data = get(symbol, retry=5)
        logger.debug(f"[get_quote] TWSE/TPEX response for {symbol}: {data}")
        if not data.get('success') or 'realtime' not in data:
            logger.error(f"[get_quote] Taiwan stock data fetch failed for {symbol}: {data.get('rtmessage', 'No data')}")
            # Fallback to Finnhub
            logger.info(f"[get_quote] Attempting Finnhub fallback for {symbol}.TW")
            data = get_finnhub_json("quote", {"symbol": f"{symbol}.TW"})
            logger.debug(f"[get_quote] Finnhub fallback response for {symbol}.TW: {data}")
            if not data or 'c' not in data:
                logger.error(f"[get_quote] Finnhub fallback failed for {symbol}.TW: {data}")
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
        logger.debug(f"[get_quote] Formatted quote for {symbol}: {quote}")
        return quote
    except Exception as e:
        logger.error(f"[get_quote] Error fetching Taiwan quote for {symbol}: {e}", exc_info=True)
        return {}

def get_metrics(symbol):
    logger.debug(f"[get_metrics] Fetching metrics for {symbol}.TW")
    try:
        metrics = get_finnhub_json("stock/metric", {"symbol": f"{symbol}.TW", "metric": "all"}).get("metric", {})
        logger.debug(f"[get_metrics] Metrics for {symbol}.TW: {metrics}")
        if not metrics:
            logger.warning(f"[get_metrics] No metrics data for {symbol}.TW")
        return metrics
    except Exception as e:
        logger.error(f"[get_metrics] Error fetching metrics for {symbol}.TW: {e}", exc_info=True)
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
    logger.debug(f"[filter_metrics] Filtered metrics: {filtered}")
    return filtered

def get_recent_news(symbol):
    logger.debug(f"[get_recent_news] Fetching news for {symbol}.TW")
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        past = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        news = get_finnhub_json("company-news", {"symbol": f"{symbol}.TW", "from": past, "to": today})
        logger.debug(f"[get_recent_news] News for {symbol}.TW: {news}")
        if not isinstance(news, list):
            logger.warning(f"[get_recent_news] No news data for {symbol}.TW")
            return []
        news = sorted(news, key=lambda x: x.get("datetime", 0), reverse=True)[:10]
        for n in news:
            try:
                n["datetime"] = datetime.datetime.utcfromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M")
            except:
                n["datetime"] = "未知時間"
        return news
    except Exception as e:
        logger.error(f"[get_recent_news] Error fetching news for {symbol}.TW: {e}", exc_info=True)
        return []

def get_company_profile(symbol):
    logger.debug(f"[get_company_profile] Fetching profile for {symbol}.TW")
    try:
        profile = get_finnhub_json("stock/profile2", {"symbol": f"{symbol}.TW"})
        logger.debug(f"[get_company_profile] Profile for {symbol}.TW: {profile}")
        if not profile:
            logger.warning(f"[get_company_profile] No profile data for {symbol}.TW")
        return profile
    except Exception as e:
        logger.error(f"[get_company_profile] Error fetching company profile for {symbol}.TW: {e}", exc_info=True)
        return {}

def calculate_rsi(series, period=14):
    logger.debug("[calculate_rsi] Calculating RSI")
    try:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rs = rs.replace([float('inf'), -float('inf')], 0)
        rsi = 100 - (100 / (1 + rs))
        logger.debug(f"[calculate_rsi] RSI calculated: {rsi.iloc[-1]}")
        return rsi.iloc[-1]
    except Exception as e:
        logger.error(f"[calculate_rsi] Error calculating RSI: {e}", exc_info=True)
        return 'N/A'

def get_historical_data(symbol):
    logger.debug(f"[get_historical_data] Fetching historical data for {symbol}")
    try:
        stock = Stock(symbol)
        today = datetime.datetime.today()
        before = today - datetime.timedelta(days=365)
        stock.fetch_from(before.year, before.month)
        logger.debug(f"[get_historical_data] Stock data for {symbol}: {stock.data}")
        if not stock.data:
            logger.warning(f"[get_historical_data] No historical data for Taiwan stock {symbol}")
            return pd.DataFrame(), {}
        data_dict = {
            'Date': stock.date,
            'Open': stock.open,
            'High': stock.high,
            'Low': stock.low,
            'Close': stock.price,
            'Volume': stock.capacity
        }
        df = pd.DataFrame(data_dict)
        df.set_index('Date', inplace=True)
        df = df.dropna(subset=['Close'])
        logger.debug(f"[get_historical_data] Historical data DataFrame for {symbol}: {df.head().to_dict()}")
        window_50 = min(50, len(df))
        ma50 = df['Close'].rolling(window=window_50).mean().iloc[-1] if len(df) >= 1 else 'N/A'
        rsi = calculate_rsi(df['Close'])
        ema12 = df['Close'].ewm(span=12, adjust=False).mean().iloc[-1] if len(df) >= 1 else 'N/A'
        ema26 = df['Close'].ewm(span=26, adjust=False).mean().iloc[-1] if len(df) >= 1 else 'N/A'
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
        logger.debug(f"[get_historical_data] Technical indicators for {symbol}: {technical}")
        return df, technical
    except Exception as e:
        logger.error(f"[get_historical_data] Error in get_historical_data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame(), {}

def get_plot_html(df, symbol, currency="TWD"):
    try:
        if df.empty or 'Close' not in df.columns:
            logger.warning(f"[get_plot_html] No data for plot for {symbol}")
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
        logger.error(f"[get_plot_html] Error generating plot for {symbol}: {e}", exc_info=True)
        return "<p class='text-danger'>📊 無法生成股價趨勢圖 / Unable to generate price trend chart</p>"

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
        logger.debug(f"[index] Processing POST request with form data: {request.form}")
        if request_count >= current_limit:
            result["error"] = f"已達 {current_tier_name} 等級請求上限，請升級方案 / Request limit reached for {current_tier_name}, please upgrade your plan"
            logger.warning(f"[index] Request limit reached: {request_count}/{current_limit}")
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        symbol = request.form.get("symbol", "").strip().upper()
        logger.debug(f"[index] Received symbol: {symbol}")
        if not symbol or not len(symbol) == 4 or not symbol.isdigit():
            result["error"] = "請輸入有效的台灣股票代號 (4位數字) / Please enter a valid Taiwan stock symbol (4 digits)"
            logger.warning(f"[index] Invalid symbol provided: {symbol}")
            return render_template("index.html", result=result, symbol_input=symbol,
                                   tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                   stripe_mode=STRIPE_MODE, request_count=request_count,
                                   current_tier_name=current_tier_name, current_limit=current_limit)

        try:
            session["request_count"] = request_count + 1
            currency = "TWD"

            quote = get_quote(symbol)
            if not quote or all(v == 'N/A' for v in quote.values()):
                result["error"] = f"無法取得 {symbol} 的即時報價資料，可能是API失敗或股票代號無效 / Unable to fetch quote data for {symbol}, possibly due to API failure or invalid symbol"
                logger.error(f"[index] No valid quote data for {symbol}")
                return render_template("index.html", result=result, symbol_input=symbol,
                                       tiers=PRICING_TIERS, stripe_pub_key=STRIPE_PUBLISHABLE_KEY,
                                       stripe_mode=STRIPE_MODE, request_count=request_count,
                                       current_tier_name=current_tier_name, current_limit=current_limit)

            metrics = filter_metrics(get_metrics(symbol))
            news = get_recent_news(symbol)
            profile = get_company_profile(symbol)
            industry_en = profile.get("finnhubIndustry", "Unknown")
            industry_zh = industry_mapping.get(industry_en, "未知")
            df, technical = get_historical_data(symbol)
            quote['volume'] = technical.get('volume', quote.get('volume', 'N/A'))
            plot_html = get_plot_html(df, symbol, currency)

            bfp_signal = "無明確信號 / No clear signal"
            try:
                stock = Stock(symbol)
                if stock.data:
                    bfp = BestFourPoint(stock)
                    best = bfp.best_four_point()
                    if best:
                        bfp_signal = f"買入信號: {best[1]} / Buy signal: {best[1]}" if best[0] else f"賣出信號: {best[1]} / Sell signal: {best[1]}"
                    logger.debug(f"[index] Best Four Point signal for {symbol}: {bfp_signal}")
                else:
                    logger.warning(f"[index] No stock data for BestFourPoint analysis for {symbol}")
            except Exception as e:
                logger.error(f"[index] Error in BestFourPoint analysis for {symbol}: {e}", exc_info=True)
                bfp_signal = "無法計算最佳四點信號 / Unable to calculate Best Four Point signal"

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
            logger.debug(f"[index] OpenAI prompt: {prompt}")
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
                logger.debug(f"[index] OpenAI response: {gpt_analysis}")
            except Exception as e:
                logger.error(f"[index] OpenAI API error for {symbol}: {e}", exc_info=True)
                gpt_analysis = {
                    'summary': (
                        f"無法生成分析，請稍後重試 / Failed to generate analysis, please try again later\n\n"
                        f"---\n\n*以上分析僅供參考，投資有風險 / The above analysis is for reference only, investment carries risks*"
                    )
                }

            result = {
                "symbol": symbol,
                "quote": {k: v for k, v in quote.items() if v != 'N/A'},
                "profile": profile,
                "industry_en": industry_en,
                "industry_zh": industry_zh,
                "metrics": metrics,
                "news": news,
                "gpt_analysis": gpt_analysis,
                "plot_html": plot_html,
                "technical": {k: v for k, v in technical.items() if v != 'N/A'},
                "currency": currency,
                "bfp_signal": bfp_signal
            }
            logger.debug(f"[index] Final result for {symbol}: {result}")
        except Exception as e:
            result = {"error": f"無法取得 {symbol} 的股票資料，可能是API連線問題或資料解析錯誤 / Unable to fetch data for {symbol}: {str(e)}"}
            logger.error(f"[index] Processing error for symbol {symbol}: {e}", exc_info=True)

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
