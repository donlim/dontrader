import os
import json
import requests
from dotenv import load_dotenv

# Load your keys from .env
load_dotenv()

HL_API_KEY = os.getenv("HL_API_KEY")
HL_API_SECRET = os.getenv("HL_API_SECRET")
HL_WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS")

def fetch_meta():
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "meta"
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def run_diagnostics():
    print("🔬 Running Hyperliquid API Diagnostics...\n")

    try:
        meta = fetch_meta()
        universe = meta.get("universe", [])

        print(f"✅ Total tradable symbols: {len(universe)}\n")
        for asset in universe:
            print(f" - {asset['name']}")

        print("\n✅ If BTCUSDC, ETHUSDC, HYPEUSDC are NOT listed above,")
        print("👉 that explains why you are not receiving l2Book data.\n")

    except Exception as e:
        print(f"❌ Error fetching meta info: {e}")

if __name__ == "__main__":
    run_diagnostics()
