import asyncio
from trading_bot.tasks.live_trader import _main as launch_all

if __name__ == "__main__":
    try:
        asyncio.run(launch_all())
    except KeyboardInterrupt:
        print("Bot manually stopped.")
