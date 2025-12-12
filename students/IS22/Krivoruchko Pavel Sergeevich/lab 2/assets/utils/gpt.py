from openai import AsyncOpenAI
from config import OPENAI_API_KEY, SYSTEM_PROMPT
from database import get_dialog_history, save_dialog_history
import logging

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def get_response(message: str, user_id: int, user_name: str, client: AsyncOpenAI) -> str:
    history = await get_dialog_history(user_id)
    history.append({"role": "user", "content": message})
    if len(history) > 6:
        history = history[-6:]
    input_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(user_name=user_name or "друг")
        }
    ] + history

    try:
        response = await client.responses.create(
            model="gpt-4.1-nano",
            input=input_messages
        )
        ai_message = response.output_text
        history.append({"role": "assistant", "content": ai_message})
        await save_dialog_history(user_id, history)
        return ai_message

    except Exception as e:
        logging.error(f"Error: {e}")
        return "Не удалось получить ответ"