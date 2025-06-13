import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(BASE_DIR, "strategy_profile.json")

with open(PROFILE_FILE, "r") as f:
    STRATEGY_PROFILES = json.load(f)

def get_profile(symbol):
    return STRATEGY_PROFILES.get(symbol, STRATEGY_PROFILES["BTC"])