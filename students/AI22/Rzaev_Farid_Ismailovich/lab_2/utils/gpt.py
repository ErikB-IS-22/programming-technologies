from openai import AsyncOpenAI
from config import API_KEY,PROXY
import httpx
import logging

http_client = httpx.AsyncClient(proxy=PROXY)
client = AsyncOpenAI(api_key=API_KEY, http_client=http_client )

async def get_response(message: str, client: AsyncOpenAI) -> str:
    try:
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=message
        )
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"