import logging
from utils.database import db
from openai import AsyncOpenAI
from config import MISTRAL_API_KEY, SYSTEM_PROMPT, MAX_HISTORY_MESSAGES


client = AsyncOpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

async def get_response(message: str,
                       user_id: int = None,
                       user_name: str = None,
                       client: AsyncOpenAI = client,
                       model: str = "mistral-small-latest",
                       use_context: bool = True) -> str:
    try:
        system_prompt = SYSTEM_PROMPT
        if user_name:
            system_prompt += f"\n\nИмя пользователя: {user_name}. Обращайся к нему по имени в своих ответах."
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if use_context and user_id:
            context = await db.get_conversation_context(user_id, MAX_HISTORY_MESSAGES)
            
            for msg in context:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({"role": "user", "content": message})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.5
        )

        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"