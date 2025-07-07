import asyncio
from trading_bot.tasks.asset_manager_64 import launch_all

if __name__ == "__main__":
    try:
        asyncio.run(launch_all())
    except KeyboardInterrupt:
        print("Bot manually stopped.")
