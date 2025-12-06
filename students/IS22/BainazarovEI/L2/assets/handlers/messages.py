from utils.loader import dp
import logging
from aiogram import F
from aiogram.types import Message
from utils.gpt import get_response

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        response = await get_response(message.text, message)
        await message.answer(response, parse_mode=None)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")

@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    await message.answer("Вы отправили картинку!")