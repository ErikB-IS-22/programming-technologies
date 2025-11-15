import os
import requests
from dotenv import load_dotenv
from chat_history_manager import save_message, get_recent_history, clear_history
from system_prompt_manager import get_active_system_prompt, manage_system_prompts
from database import init_db

load_dotenv()

def get_response(user_input):
    url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
    
    system_prompt = get_active_system_prompt(db_manager)
    history = get_recent_history(db_manager)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": messages,
        "temperature": 0.01
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

if __name__ == "__main__":
    db_manager = init_db()
    
    print("=== Чат с DeepSeek-R1 ===")
    
    while True:
        print("\nГлавное меню:")
        print("1. Начать общение")
        print("2. Управление системными промптами")
        print("3. Очистить историю диалога")
        print("4. Выйти")
        
        choice = input("Выберите действие: ")
        
        match choice:
            case "1":
                print("\nРежим общения (введите 'back' для возврата в меню):")
                while True:
                    question = input("Вы: ")
                    if question.lower() == "back":
                        break
                    if question.lower() == "exit":
                        print("Завершение программы.")
                        exit()
                    answer = get_response(question)
                    print("AI:", answer)
            case "2":
                manage_system_prompts(db_manager)
            case "3":
                clear_history()
                print("История диалога очищена!")
            case "4":
                print("Завершение программы.")
                break
            case _:
                print("Неверный выбор. Попробуйте снова.")