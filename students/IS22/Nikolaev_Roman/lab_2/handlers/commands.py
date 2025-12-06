import logging

from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from utils.database import db_manager
from utils.loader import dp


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await message.answer(
            f"Привет, {message.from_user.full_name}, я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них.\nПожалуйста, помни про свой баланс на счету аккаунта в OpenAI и не ддось меня без необходимости)"
        )
    except Exception as e:
        logging.error(f"Error occurred: {e}")


@dp.message(Command("reset_context"))
async def command_reset_context_handler(message: Message) -> None:
    try:
        user_id = message.from_user.id
        db_manager.reset_user_context(user_id)
        await message.answer("Контекст диалога успешно сброшен.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при сбросе контекста диалога.")
