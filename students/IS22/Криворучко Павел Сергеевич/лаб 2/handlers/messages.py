from utils.loader import dp
import logging
from aiogram.types import Message
from utils.gpt import get_response, client

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "друг"
        if message.text:
            response = await get_response(message.text, user_id, user_name, client)
            await message.answer(response)
        elif message.photo:
            await message.answer(f"Извините, но я не работаю с фотографиями")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")