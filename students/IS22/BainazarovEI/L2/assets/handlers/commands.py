from utils.loader import dp
import logging
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from database import clear_history, init_db

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await message.answer(f"Привет, {message.from_user.full_name}, я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них. \
            Пожалуйста, помни про свой баланс на счету аккаунта и не ддось меня без необходимости)")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

@dp.message(Command("resetcontext"))
async def cmd_reset(message: Message):
    db_manager = init_db()
    try:
        clear_history(db_manager)
        await message.answer("Контекст разговора сброшен. Начинаем новый диалог!")
    except Exception as e:
        logging.error(f"Error occurred: {e}")