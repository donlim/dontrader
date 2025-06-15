# trading_bot/config/exchange_config.py

import os
from dotenv import load_dotenv

# Load from .env automatically
load_dotenv()

# === Trading Mode ===
TRADING_MODE = 'paper'  # 'paper', 'dry-run', 'live'

# === Hyperliquid API Credentials ===
HYPERLIQUID_WALLET_ADDRESS = os.getenv('HL_WALLET_ADDRESS')
HYPERLIQUID_API_SECRET = os.getenv('HL_API_SECRET')

# === Hyperliquid API Endpoint ===
HYPERLIQUID_BASE_URL = 'https://api.hyperliquid.xyz'