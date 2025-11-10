from utils.loader import dp
import logging
from aiogram.types import Message
from utils.gpt import get_response, client

@dp.message()
async def message_handler(message: Message) -> None:
    try:
        if message.text:
            response = await get_response(message.text, message.from_user.id, message.from_user.full_name, client)
            await message.answer(response)
        else:
            await message.answer(f"Спасибо, что отправили {str(message.content_type)[12:]}, но я работаю только с обычными сообщениями)")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")
