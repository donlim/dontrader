# trading_bot/config/config.py

import os
from dotenv import load_dotenv

load_dotenv()

HL_API_KEY = os.getenv("HL_API_KEY")
HL_API_SECRET = os.getenv("HL_API_SECRET")

# Use hyperliquid diagnostic output symbols (ex: BTC, ETH, not BTCUSDC)
SYMBOLS = ["BTC", "ETH", "HYPE"]

# Add your websocket endpoint here
WS_URL = "wss://api.hyperliquid.xyz/ws"