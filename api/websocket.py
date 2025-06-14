# trading_bot/api/websocket.py

import asyncio
import websockets
import json
from trading_bot.config.config import SYMBOLS, WS_URL
from trading_bot.state.buffers import update_buffer, orderbooks

async def subscribe(ws, symbol):
    # BookTop subscription (best bid/ask)
    await ws.send(json.dumps({
        "method": "subscribe",
        "subscription": {
            "type": "bookTop",
            "coin": symbol
        }
    }))
    # Trades subscription (for volume)
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

            if data["channel"] == "bookTop":
                coin = data["data"]["coin"]
                best_bid = float(data["data"]["levels"][0][0])
                best_ask = float(data["data"]["levels"][1][0])
                mid_price = (best_bid + best_ask) / 2

                bid_size = float(data["data"]["levels"][0][1])
                ask_size = float(data["data"]["levels"][1][1])
                imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-6)

                orderbooks[coin].update(mid_price, imbalance)
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