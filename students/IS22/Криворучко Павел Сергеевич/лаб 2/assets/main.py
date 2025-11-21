import asyncio
import logging
import sys
from handlers import dp
from utils.loader import bot
from database import init_db

async def main():
    try:
        await init_db()   
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())