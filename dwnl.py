import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np


class DataLoader:
    ASSETS = {
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "GOLD": "GC=F",
        "OIL": "CL=F",
        "EURUSD": "EURUSD=X",
        "BTC": "BTC-USD",
        "AAPL": "AAPL",
        "JPM": "JPM",
    }

    def download(
        self, ticker: str = "^GSPC", start: str = "2010-01-01", end: str = "2024-12-31"
    ) -> dict:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)["Close"]
        data = data.ffill().dropna()
        log_ret = np.log(data / data.shift(1)).dropna()
        log_ret.name = ticker

        rv = (log_ret**2).rename("realized_var")

        return {
            "prices": data,
            "returns": log_ret,
            "realized_var": rv,
        }

    def get_vix(self, start: str = "2010-01-01", end: str = "2024-12-31") -> pd.Series:
        try:
            vix = yf.download("^VIX", start=start, end=end, auto_adjust=True)["Close"]
            vix.name = "VIX"
            return vix.ffill().dropna()
        except:
            return web.DataReader("VIXCLS", "fred", start, end).squeeze()

    def split(self, series: pd.Series, train_frac=0.70, val_frac=0.15) -> tuple:
        T = len(series)
        t1 = int(T * train_frac)
        t2 = int(T * (train_frac + val_frac))
        return series.iloc[:t1], series.iloc[t1:t2], series.iloc[t2:]
