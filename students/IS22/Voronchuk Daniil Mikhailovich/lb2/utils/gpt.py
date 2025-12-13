from openai import AsyncOpenAI
from config import OPENAI_API_KEY, SYSTEM_PROMPT
import logging

from utils.db import save_message, get_user_messages

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def get_response(message: str, client: AsyncOpenAI, user_id: int, user_name: str = "") -> str:
    try:

        save_message(user_id, "user", message)
        

        history = get_user_messages(user_id)
        context_text = "\n".join([f"{role}: {content}" for role, content in history])
        
        prompt = f"{SYSTEM_PROMPT}\nПользователь: {user_name}\n{context_text}\nAI:"
        
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        reply = response.output_text
        
        save_message(user_id, "assistant", reply)
        return reply
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"
