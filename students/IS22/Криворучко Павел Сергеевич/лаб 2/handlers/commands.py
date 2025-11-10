from utils.loader import dp
import logging
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from utils.gpt import user_histories

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await message.answer(f"Привет, {message.from_user.full_name}, я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них. \
            Пожалуйста, помни про свой баланс на счету аккаунта в OpenAI и не ддось меня без необходимости)")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

@dp.message(Command("resetcontext"))
async def reset_context(message: Message):
    user_id = message.from_user.id

    if user_id in user_histories:
        user_histories[user_id] = [] 
        await message.answer("История диалога сброшена. Можешь начинать новый разговор!")
    else:
        await message.answer("История не была найдена. Начни новый разговор!")