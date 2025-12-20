import json
import logging
import os

from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from utils.database import db_manager
from utils.loader import dp


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        user_name = message.from_user.full_name if message.from_user else "Пользователь"
        user_id = message.from_user.id if message.from_user else 0

        welcome_text = (
            f"Привет, {user_name}! Я твой бот-ассистент.\n"
            "Можешь задавать мне вопросы, и я буду отвечать на них.\n"
            "Пожалуйста, помни про свой баланс на счету OpenAI и не перегружай меня запросами без необходимости."
        )

        await message.answer(welcome_text)

        history = db_manager.get_last_messages(user_id=user_id, limit=5)

        if history:
            os.makedirs("data_json", exist_ok=True)

            file_path = f"data_json/history_{user_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"Error occurred: {e}")


@dp.message(Command("reset_context"))
async def command_reset_context_handler(message: Message) -> None:
    try:
        user_id = message.from_user.id
        db_manager.clear_history(user_id)
        file_path = f"data_json/history_{user_id}.json"
        if os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

        await message.answer("Контекст диалога успешно сброшен.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при сбросе контекста диалога.")
