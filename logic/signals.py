# trading_bot/logic/signals.py

# ✅ This is where you plug your model rules
def generate_signal(indicators):
    delta_flow = indicators.get("DELTA_FLOW", 0)
    book_imb = indicators.get("BOOK_IMB", 0)
    price = indicators.get("PRICE", 0)

    # Sample rule: very simple delta flow trigger
    if delta_flow > 5 and book_imb > 0.7:
        return 'BUY'
    elif delta_flow < -5 and book_imb < -0.7:
        return 'SELL'
    else:
        return 'HOLD'