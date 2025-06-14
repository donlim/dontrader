# trading_bot/api/websocket.py

import asyncio
import websockets
import json
from trading_bot.config.config import SYMBOLS, WS_URL
from trading_bot.state.buffers import update_buffer, orderbooks

async def subscribe(ws, symbol):
    # Subscribe to full depth l2Book
    await ws.send(json.dumps({
        "method": "subscribe",
        "subscription": {
            "type": "l2Book",
            "coin": symbol
        }
    }))
    # Subscribe to trades for volume
    await ws.send(json.dumps({
        "method": "subscribe",
        "subscription": {
            "type": "trades",
            "coin": symbol
        }
    }))

async def stream_symbol(symbol):
    async with websockets.connect(WS_URL) as ws:
        await subscribe(ws, symbol)
        print(f"Authenticated websocket for: {symbol}")

        while True:
            message = await ws.recv()
            data = json.loads(message)

            if data["channel"] == "l2Book":
                coin = data["data"]["coin"]
                bids_raw, asks_raw = data["data"]["levels"]

                # Parse bids and asks
                bids = [(float(level["px"]), float(level["sz"])) for level in bids_raw]
                asks = [(float(level["px"]), float(level["sz"])) for level in asks_raw]

                # Full depth imbalance calculation
                bid_vol = sum(size for _, size in bids)
                ask_vol = sum(size for _, size in asks)
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

                mid_price = (bids[0][0] + asks[0][0]) / 2

                # ✅ Forward to buffers + orderbooks
                orderbooks[coin].update(mid_price, imbalance, bids, asks)
                update_buffer(coin, mid_price, None)

            elif data["channel"] == "trades":
                for trade in data["data"]:
                    coin = trade["coin"]
                    price = float(trade["px"])
                    size = float(trade["sz"])
                    update_buffer(coin, price, size)

            else:
                print(f"Unknown message for {symbol}: {data}")

async def handle_websocket():
    await asyncio.gather(*[stream_symbol(symbol) for symbol in SYMBOLS])