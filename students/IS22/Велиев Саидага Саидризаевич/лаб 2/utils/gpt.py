from openai import AsyncOpenAI
from config import api_key, prompt
import logging
from utils.db_connect import save_dialog_history, get_dialog_history


client = AsyncOpenAI(api_key=api_key)

async def get_response(message: str, user_id: str, user_name: str, client: AsyncOpenAI) -> str:
    dialog_history_actual = get_dialog_history(user_id)
    dialog_history_actual.append({"role": "user", "content": message})
    if len(dialog_history_actual) > 50:
        dialog_history_actual.pop(0)

    try:
        response = await client.responses.create(
            model="gpt-4.1-nano",
            input=[{"role": "system", "content": prompt}] + [{"role": "user", "content": f"Пользователя зовут {user_name}"}] + dialog_history_actual,
        )
        dialog_history_actual.append({"role": "assistant", "content": response.output_text})
        save_dialog_history(user_id, dialog_history_actual)
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"
