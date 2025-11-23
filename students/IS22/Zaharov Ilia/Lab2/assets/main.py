import sys
import asyncio
import logging
from handlers import dp
from utils.loader import bot
from utils.database import db


async def main():
    try:
        await db.create_tables()
        logging.info("Database initialized successfully")
        
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Error occured: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout)
    asyncio.run(main())