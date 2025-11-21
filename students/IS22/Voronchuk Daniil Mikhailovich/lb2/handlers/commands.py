from utils.loader import dp
import logging
from aiogram.filters import Command
from aiogram.types import Message
from utils.db import reset_user_context

@dp.message(Command("reset-context"))
async def reset_context_handler(message: Message):
    try:
        reset_user_context(message.from_user.id)
        await message.answer("Контекст диалога сброшен!")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Не удалось сбросить контекст.")

    