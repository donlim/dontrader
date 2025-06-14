# trading_bot/utils/reconnect.py

import asyncio
from trading_bot.utils.logging import logger

async def resilient_stream(stream_fn, symbol):
    while True:
        try:
            await stream_fn(symbol)
        except Exception as e:
            logger.warning(f"WebSocket error for {symbol}: {e}")
            await asyncio.sleep(5)  # backoff before reconnect