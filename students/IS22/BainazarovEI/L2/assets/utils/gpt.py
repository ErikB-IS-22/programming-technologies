import os
import requests
from config import DEEPSEEK_API_KEY
from aiogram.types import Message
from database import get_recent_history, save_message, init_db

async def get_response(user_input, message: Message):
    db_manager = init_db()
    
    url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
    
    history = get_recent_history(db_manager)

    if message and message.from_user:
        user = message.from_user.full_name or message.from_user.first_name or "пользователь"
    else:
        user = "пользователь"

    messages = [{"role": "system", "content": f"You are a helpful assistant. Always address the user by name before replying. The person you answer to is {user}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    data = response.json()

    if 'choices' in data and len(data['choices']) > 0:
        text = data['choices'][0]['message']['content']
        
        save_message(db_manager, "user", user_input)
        
        if '</think>' in text:
            assistant_response = text.split('</think>')[1].strip()
        else:
            assistant_response = text.strip()
        
        save_message(db_manager, "assistant", assistant_response)
        
        return assistant_response
    else:
        return "Ошибка: не удалось получить ответ от API"