# trading_bot/execution/exchange_adapter.py

"""
4.0 - Hyperliquid Exchange Adapter (Fully Unified: Paper/Live Only)
"""

import requests
import json
import hmac
import hashlib
from trading_bot.config import exchange_config

# === Load config ===
TRADING_MODE = exchange_config.TRADING_MODE
HL_WALLET_ADDRESS = exchange_config.HYPERLIQUID_WALLET_ADDRESS
HL_API_SECRET = exchange_config.HYPERLIQUID_API_SECRET
HL_BASE_URL = exchange_config.HYPERLIQUID_BASE_URL

# === Master order function ===

def submit_order(symbol, side, size, price):
    """
    Main external trade function to place orders live.
    Paper mode handled entirely by paper_engine.
    """

    if TRADING_MODE == 'paper':
        # Paper trades handled fully inside paper_engine
        print(f"[PAPER ENGINE HANDLES PAPER MODE] {side} {size:.4f} {symbol} @ {price:.2f}")
        return

    elif TRADING_MODE == 'live':
        place_live_order(symbol, side, size, price)
        return

    else:
        raise ValueError(f"Unknown TRADING_MODE: {TRADING_MODE}")

# === LIVE ORDER PLACEMENT ===

def place_live_order(symbol, side, size, price):
    """
    This submits real order to Hyperliquid API.
    """

    path = "/api/v1/order"
    url = HL_BASE_URL + path

    payload = {
        "wallet": HL_WALLET_ADDRESS,
        "symbol": symbol,
        "side": side,
        "size": size,
        "price": price,
        "type": "limit",
        "timeInForce": "gtc"
    }

    payload_str = json.dumps(payload, separators=(',', ':'))
    signature = hmac.new(
        bytes.fromhex(HL_API_SECRET[2:]),  # strip '0x'
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'Content-Type': 'application/json',
        'HL-SIGNATURE': signature
    }

    response = requests.post(url, headers=headers, data=payload_str)

    if response.status_code == 200:
        print(f"[LIVE] Order placed successfully: {side} {size} {symbol} @ {price}")
    else:
        print(f"[LIVE ERROR] Status: {response.status_code} | Response: {response.text}")