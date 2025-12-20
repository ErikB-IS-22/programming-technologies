import logging

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, SYSTEM_PROMPT, TEMPERATURE

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_response(
    message: str, username: str, client: AsyncOpenAI, history_json: list[dict]
) -> str:
    try:
        personalized_prompt = (
            SYSTEM_PROMPT.format(username=username)
            if "{username}" in SYSTEM_PROMPT
            else f"{SYSTEM_PROMPT} Общайся с пользователем по имени {username}."
        )

        last_history = history_json[-5:] if history_json else []

        context_text = []
        for item in last_history:
            user_msg = item.get("user_message", "")
            assistant_msg = item.get("assistant_message", "")
            if user_msg:
                context_text += f"Пользователь: {user_msg}\n"
            if assistant_msg:
                context_text += f"Ассистент: {assistant_msg}\n"

        if context_text:
            full_message = f"Контекст диалога:\n{context_text}\n\nТекущий запрос пользователя :\n{message}"
        else:
            full_message = message

        response = await client.responses.create(
            model="gpt-4.1-nano",
            input=full_message,
            instructions=personalized_prompt,
            temperature=TEMPERATURE,
        )
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"
