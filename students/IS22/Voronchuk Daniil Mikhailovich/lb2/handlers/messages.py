from utils.loader import dp, client
from aiogram.types import Message
from utils.gpt import get_response
from utils.db import save_message
from utils.loader import dp, client

import logging

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        user_id = message.from_user.id
        user_name = message.from_user.full_name

        if message.photo:
            await message.answer("Вы отправили картинку!")
            save_message(user_id, "user", "[Фото]")
            save_message(user_id, "assistant", "Вы отправили картинку!")
            return

        response = await get_response(message.text, client, user_id, user_name)
        await message.answer(response)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")