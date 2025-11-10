from openai import AsyncOpenAI
from config import OPENAI_API_KEY, SYSTEM_PROMPT
import logging

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

user_histories = {}

async def get_response(message: str, user_id: int, user_name: str, client: AsyncOpenAI) -> str:
    if user_id not in user_histories:
        user_histories[user_id] = []

    history = user_histories[user_id]
    history.append({"role": "user", "content": message})

    if len(history) > 6:
        history.pop(0)

    input_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(user_name=user_name or "друг")}] + history

    try:
        response = await client.responses.create(
            model="gpt-4.1-nano",
            input=input_messages
        )
        ai_message = response.output_text
        history.append({"role": "assistant", "content": ai_message})
        return ai_message
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"