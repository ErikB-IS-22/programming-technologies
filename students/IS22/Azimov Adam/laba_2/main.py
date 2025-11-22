import asyncio
import logging
import sys
from utils.loader import bot, dp
from handlers.commands import dp
from handlers.messages import dp


async def main():
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
