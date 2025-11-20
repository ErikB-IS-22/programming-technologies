from utils.loader import dp
import logging
from aiogram.types import Message
from utils.gpt import get_response, client

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        response = await get_response(message.text, client)
        await message.answer(response)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")