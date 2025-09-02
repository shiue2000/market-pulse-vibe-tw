# -*- coding: utf-8 -*-

import datetime
import urllib.parse
from collections import namedtuple
import logging

from twstock.proxy import get_proxies

try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

import requests

try:
    from . import analytics
    from .codes import codes
except ImportError as e:
    if e.name == "lxml":
        raise e
    import analytics
    from codes import codes

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    def __init__(self):
        pass

    def fetch(self, year: int, month: int, sid: str, retry: int = 5):
        params = {"date": "%d%02d01" % (year, month), "stockNo": sid}
        for retry_i in range(retry):
            try:
                r = requests.get(self.REPORT_URL, params=params, proxies=get_proxies(), timeout=15)
                r.raise_for_status()
                data = r.json()
                logger.debug(f"[TWSE Fetch] Response for {sid}: {data}")
                if data["stat"] == "OK":
                    data["data"] = self.purify(data)
                    return data
                logger.warning(f"[TWSE Fetch] Non-OK stat for {sid}: {data.get('stat')}")
            except (JSONDecodeError, requests.RequestException) as e:
                logger.warning(f"[TWSE Fetch Error] Attempt {retry_i + 1} for {sid}: {e}")
                time.sleep(5)
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
    REPORT_URL = urllib.parse.urljoin(
        TPEX_BASE_URL, "web/stock/aftertrading/daily_trading_info/st43_result.php"
    )

    def __init__(self):
        pass

    def fetch(self, year: int, month: int, sid: str, retry: int = 5):
        params = {"d": "%d/%d" % (year - 1911, month), "stkno": sid}
        for retry_i in range(retry):
            try:
                r = requests.get(self.REPORT_URL, params=params, proxies=get_proxies(), timeout=15)
                r.raise_for_status()
                data = r.json()
                logger.debug(f"[TPEX Fetch] Response for {sid}: {data}")
                data["data"] = self.purify(data) if data.get("aaData") else []
                return data
            except (JSONDecodeError, requests.RequestException) as e:
                logger.warning(f"[TPEX Fetch Error] Attempt {retry_i + 1} for {sid}: {e}")
                time.sleep(5)
        logger.error(f"[TPEX Fetch Error] Failed to fetch data for {sid} after {retry} attempts")
        return {"aaData": [], "data": []}

    def _convert_date(self, date):
        """Convert '106/05/01' to '2017/05/01'"""
        return "/".join([str(int(date.split("/")[0]) + 1911)] + date.split("/")[1:])

    def _make_datatuple(self, data):
        try:
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
        except Exception as e:
            logger.error(f"[TPEX Parse Error] Failed to parse data: {e}")
            return None

    def purify(self, original_data):
        return [d for d in (self._make_datatuple(d) for d in original_data["aaData"]) if d is not None]

class Stock(analytics.Analytics):
    def __init__(self, sid: str, initial_fetch: bool = True):
        self.sid = sid
        try:
            market = codes[sid].market if hasattr(codes[sid], 'market') else codes[sid].get('market', '上市')
            logger.debug(f"[Stock Init] Market for {sid}: {market}")
            self.fetcher = TWSEFetcher() if market == "上市" else TPEXFetcher()
        except Exception as e:
            logger.error(f"[Stock Init Error] Failed to determine market for {sid}: {e}", exc_info=True)
            raise
        self.raw_data = []
        self.data = []

        # Init data
        if initial_fetch:
            self.fetch_31()

    def _month_year_iter(self, start_month, start_year, end_month, end_year):
        ym_start = 12 * start_year + start_month - 1
        ym_end = 12 * end_year + end_month
        for ym in range(ym_start, ym_end):
            y, m = divmod(ym, 12)
            yield y, m + 1

    def fetch(self, year: int, month: int):
        """Fetch year month data"""
        try:
            self.raw_data = [self.fetcher.fetch(year, month, self.sid)]
            self.data = self.raw_data[0]["data"]
            logger.info(f"[Stock Fetch] Successfully fetched data for {self.sid}")
        except Exception as e:
            logger.error(f"[Stock Fetch Error] Failed for {self.sid}: {e}", exc_info=True)
            self.raw_data = [{"data": []}]
            self.data = []
        return self.data

    def fetch_from(self, year: int, month: int):
        """Fetch data from year, month to current year month data"""
        self.raw_data = []
        self.data = []
        today = datetime.datetime.today()
        for year, month in self._month_year_iter(month, year, today.month, today.year):
            self.raw_data.append(self.fetcher.fetch(year, month, self.sid))
            self.data.extend(self.raw_data[-1]["data"])
        return self.data

    def fetch_31(self):
        """Fetch 31 days data"""
        today = datetime.datetime.today()
        before = today - datetime.timedelta(days=60)
        self.fetch_from(before.year, before.month)
        self.data = self.data[-31:]
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
