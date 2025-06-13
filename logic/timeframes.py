import pandas as pd
import time

def get_current_minute_bucket():
    return int(time.time() // 60)

ohlc_store = {}

def update_ohlc(symbol, price, timestamp):
    bucket = get_current_minute_bucket()
    ohlc_store.setdefault(symbol, {}).setdefault(bucket, []).append(price)

def get_ohlc(symbol, bucket):
    prices = ohlc_store.get(symbol, {}).get(bucket, [])
    if not prices:
        return None
    series = pd.Series(prices)
    return {
        "first": series.iloc[0],
        "max": series.max(),
        "min": series.min(),
        "last": series.iloc[-1]
    }
