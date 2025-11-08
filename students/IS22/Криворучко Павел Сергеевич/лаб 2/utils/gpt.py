from openai import AsyncOpenAI
from config import OPENAI_API_KEY
import logging

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def get_response(message: str, client: AsyncOpenAI) -> str:
    try:
        response = await client.responses.create(
            model="gpt-5-nano",
            input=message
        )
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"