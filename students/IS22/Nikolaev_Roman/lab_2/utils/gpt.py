import logging

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, SYSTEM_PROMPT

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_response(
    message: str, username: str, context_data: str, client: AsyncOpenAI
) -> str:
    try:
        personalized_prompt = (
            SYSTEM_PROMPT.format(username=username)
            if "{username}" in SYSTEM_PROMPT
            else f"{SYSTEM_PROMPT} Общайся с пользователем по имени {username}."
        )

        if context_data:
            full_message = f"Контекст диалога:\n{context_data}\n\nТекущий запрос пользователя:\n{message}"
        else:
            full_message = message

        response = await client.responses.create(
            model="gpt-4.1-nano",
            input=full_message,
            instructions=personalized_prompt,
            temperature=0.8,
        )
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"
