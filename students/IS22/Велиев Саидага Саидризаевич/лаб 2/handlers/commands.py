from utils.loader import dp
import logging
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from utils.db_connect import save_dialog_history, get_dialog_history

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await message.answer(f"Привет, {message.from_user.full_name}, я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них. \nПожалуйста, помни про свой баланс на счету аккаунта в OpenAI и не ддось меня без необходимости)")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

@dp.message(Command("resetcontext"))
async def reset_context(message: Message):
    user_id = message.from_user.id  # Получаем user_id из сообщения

    # Получаем историю диалога из базы данных
    dialog_history_actual = get_dialog_history(user_id)

    if dialog_history_actual:  # Если история найдена, сбрасываем её
        dialog_history_actual = []  # Очищаем историю для данного пользователя
        save_dialog_history(user_id, dialog_history_actual)  # Сохраняем пустую историю в БД
        await message.answer("История диалога сброшена. Можешь начинать новый разговор!")
    else:
        await message.answer("История не была найдена. Начни новый разговор!")
