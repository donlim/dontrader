# trading_bot/api/websocket.py

import asyncio
import websockets
import json
from trading_bot.config.config import SYMBOLS, WS_URL
from trading_bot.state.buffers import update_buffer, orderbooks
from trading_bot.state.book_features import book_feature_buffers

# Maintain midprice snapshot per symbol for tick-rule delta flow
local_midprice = {symbol: None for symbol in SYMBOLS}

async def subscribe(ws, symbol):
    await ws.send(json.dumps({
        "method": "subscribe",
        "subscription": {"type": "l2Book", "coin": symbol}
    }))
    await ws.send(json.dumps({
        "method": "subscribe",
        "subscription": {"type": "trades", "coin": symbol}
    }))

async def process_l2book(symbol, data):
    bids_raw, asks_raw = data["levels"]

    bids = [(float(level["px"]), float(level["sz"])) for level in bids_raw]
    asks = [(float(level["px"]), float(level["sz"])) for level in asks_raw]

    bid_vol = sum(size for _, size in bids)
    ask_vol = sum(size for _, size in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
    mid_price = (bids[0][0] + asks[0][0]) / 2

    local_midprice[symbol] = mid_price

    orderbooks[symbol].update(mid_price, imbalance, bids, asks)
    update_buffer(symbol, mid_price, None)

async def process_trades(symbol, data):
    for trade in data:
        price = float(trade["px"])
        size = float(trade["sz"])

        update_buffer(symbol, price, size)

        mid_price = local_midprice.get(symbol)
        if mid_price is not None:
            if price > mid_price:
                book_feature_buffers[symbol].update_delta(buy=size, sell=0)
            elif price < mid_price:
                book_feature_buffers[symbol].update_delta(buy=0, sell=size)
            # Neutral prints at midprice: we ignore
        # If no midprice yet, skip

async def stream_symbol(symbol):
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await subscribe(ws, symbol)
                print(f"✅ Subscribed websocket for: {symbol}")

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    if data["channel"] == "l2Book":
                        await process_l2book(symbol, data["data"])
                    elif data["channel"] == "trades":
                        await process_trades(symbol, data["data"])
                    elif data["channel"] == "subscriptionResponse":
                        pass  # Optional: suppress subscription confirmations
                    else:
                        print(f"Unknown message for {symbol}: {data}")

        except Exception as e:
            print(f"⚠ Websocket error for {symbol}: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)

async def handle_websocket():
    await asyncio.gather(*[stream_symbol(symbol) for symbol in SYMBOLS])