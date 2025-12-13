import json
import logging
import os

from aiogram.types import Message

from utils.database import db_manager
from utils.gpt import client, get_response
from utils.loader import dp


@dp.message()
async def message_handler(message: Message) -> None:
    try:
        if message.photo:
            await message.answer(
                "Вы отправили картинку! К сожалению, я не могу обрабатывать изображения."
            )
            return

        username = message.from_user.full_name
        user_id = message.from_user.id

        history_file = f"data_json/history_{user_id}.json"
        os.makedirs("data_json", exist_ok=True)

        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history_json = json.load(f)
        else:
            history_json = []

        history = db_manager.get_last_messages(user_id=user_id, limit=5)

        response = await get_response(message.text, username, client, history_json)

        db_manager.add_message(user_id, username, message.text, response)

        history_json.append(
            {"user_message": message.text, "assistant_message": response}
        )
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_json, f, ensure_ascii=False, indent=2)

        await message.answer(response)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")
