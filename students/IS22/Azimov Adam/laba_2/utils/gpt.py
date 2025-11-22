from openai import AsyncOpenAI
from config import gpt_api, model
import logging

client = AsyncOpenAI(api_key=gpt_api)


async def get_response(message: str, client: AsyncOpenAI) -> str:
    try:
        response = await client.responses.create(
            model=model,
            input=message
        )
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"
