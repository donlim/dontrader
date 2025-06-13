import asyncio, websockets, json, hmac, hashlib, time
from trading_bot.config.config import HL_API_KEY, HL_API_SECRET, SYMBOLS
from trading_bot.state.buffers import update_buffer
from trading_bot.logic.orderbook import OrderbookTracker

orderbooks = {symbol: OrderbookTracker() for symbol in SYMBOLS}

async def authenticate(ws):
    timestamp = str(int(time.time() * 1000))
    signature = hmac.new(HL_API_SECRET.encode(), timestamp.encode(), hashlib.sha256).hexdigest()
    await ws.send(json.dumps({"method": "auth", "apiKey": HL_API_KEY, "timestamp": timestamp, "signature": signature}))
    print("Authenticated with Hyperliquid WebSocket ✅")

async def send_pings(ws):
    while True:
        await asyncio.sleep(30)
        await ws.send(json.dumps({"method": "ping"}))

async def handle_websocket():
    uri = "wss://api.hyperliquid.xyz/ws"
    async with websockets.connect(uri) as ws:
        await authenticate(ws)
        for symbol in SYMBOLS:
            await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": symbol}}))
            print(f"Subscribed to: {symbol}")
        asyncio.create_task(send_pings(ws))
        while True:
            try:
                data = json.loads(await ws.recv())
                if data.get("channel") == "l2Book":
                    update = data["data"]
                    symbol = update["coin"]

                    bid = float(update["levels"][0][0]["px"])
                    bid_size = float(update["levels"][0][0]["sz"])
                    ask = float(update["levels"][1][0]["px"])
                    ask_size = float(update["levels"][1][0]["sz"])
                    mid = (bid + ask) / 2
                    timestamp = update.get("time", None)

                    update_buffer(symbol, mid, timestamp)
                    orderbooks[symbol].update(bid, bid_size, ask, ask_size)
                    
                    # Live print
                    print(f"[{symbol}] Mid Price: {mid} | Imbalance: {orderbooks[symbol].get_imbalance():.3f} at {timestamp}")

            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(1)
