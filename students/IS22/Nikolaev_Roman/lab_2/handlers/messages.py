import logging

from aiogram.types import Message

from utils.gpt import client, get_response
from utils.loader import dp
from utils.database import db_manager


@dp.message()
async def message_handler(message: Message) -> None:
    try:
        if message.photo:
            await message.answer(
                "Вы отправили картинку! К сожалению, я не могу обрабатывать изображения."
            )
            return

        username = message.from_user.full_name if message.from_user else "Пользователь"
        user_id = message.from_user.id if message.from_user else 0

        context_data = db_manager.get_user_context(user_id)

        response = await get_response(message.text, username, context_data, client)

        db_manager.add_message(user_id, username, message.text, response)

        dialog_history = f"Пользователь: {message.text}\nАссистент: {response}"
        if context_data:
            combined_context = f"{context_data}\n{dialog_history}"
            if len(combined_context) > 2000:
                combined_context = combined_context[-2000:]
            db_manager.update_user_context(user_id, combined_context)
        else:
            db_manager.update_user_context(user_id, dialog_history)

        await message.answer(response)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")
