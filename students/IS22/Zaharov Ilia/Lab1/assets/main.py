import os
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

# Инициализация клиента
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1"
)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Сохранение промта в базу
def save_prompt_to_db(name, prompt_text):
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO prompts (name, prompt_text) VALUES (?, ?)', (name, prompt_text))
    conn.commit()
    conn.close()

# Получение всех промтов из базы
def get_all_prompts():
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, prompt_text FROM prompts ORDER BY created_at DESC')
    prompts = cursor.fetchall()
    conn.close()
    return prompts

# Выбор системного промта
def select_system_prompt():
    env_prompt = os.getenv("SYSTEM_PROMPT")
    if env_prompt:
        print(f"✅ Найден системный промт из .env: {env_prompt[:50]}...")
        return env_prompt
    
    print("\n" + "="*50)
    print("🤖 Выбор системного промта")
    print("1. Использовать стандартный промт")
    print("2. Выбрать из сохраненных")
    print("3. Создать новый")
    print("4. Продолжить без системного промта")
    
    choice = input("\nВыберите вариант (1-4): ").strip()
    
    if choice == "1":
        prompt = "Ты - полезный AI-ассистент. Отвечай на вопросы точно и вежливо."
        save_option = input("Сохранить этот промт в базу? (y/n): ").strip().lower()
        if save_option == 'y':
            name = input("Введите название для промта: ").strip()
            save_prompt_to_db(name, prompt)
        return prompt
    
    elif choice == "2":
        prompts = get_all_prompts()
        if not prompts:
            print("❌ Нет сохраненных промтов. Создайте новый.")
            return select_system_prompt()
        
        print("\nСохраненные промты:")
        for i, (id, name, text) in enumerate(prompts, 1):
            print(f"{i}. {name}: {text[:50]}...")
        
        try:
            prompt_num = int(input("\nВыберите номер промта: ")) - 1
            if 0 <= prompt_num < len(prompts):
                return prompts[prompt_num][2]
            else:
                print("❌ Неверный номер")
                return select_system_prompt()
        except ValueError:
            print("❌ Введите число")
            return select_system_prompt()
    
    elif choice == "3":
        print("Введите новый системный промт:")
        prompt = input().strip()
        if prompt:
            name = input("Введите название для промта: ").strip()
            save_prompt_to_db(name, prompt)
            return prompt
        else:
            print("❌ Промт не может быть пустым")
            return select_system_prompt()
    
    elif choice == "4":
        return None
    
    else:
        print("❌ Неверный выбор")
        return select_system_prompt()

def get_response(text: str, client: OpenAI, model: str = "mistral-tiny", 
                system_prompt: str = None, chat_history: list = None):
    try:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": text})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=1.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка: {e}"

# Функция для ограничения истории до 6 сообщений
def limit_history(history: list, max_messages: int = 6) -> list:
    """
    Ограничивает историю до указанного количества сообщений.
    Сохраняет баланс между сообщениями пользователя и ассистента.
    """
    if len(history) <= max_messages:
        return history
    
    return history[-max_messages:]

# Функция для показа текущей истории
def show_history(chat_history: list):
    if not chat_history:
        print("История диалога пуста")
        return
    
    print("\n" + "="*50)
    print("📜 История диалога:")
    for i, message in enumerate(chat_history, 1):
        role = "Вы" if message["role"] == "user" else "Mistral"
        print(f"{i}. {role}: {message['content'][:100]}{'...' if len(message['content']) > 100 else ''}")
    print(f"Всего сообщений: {len(chat_history)}")
    print("="*50)

if __name__ == "__main__":
    init_db()
    
    system_prompt = select_system_prompt()
    
    print("=" * 50)
    print("🤖 Mistral AI Chat")
    print("Доступные модели: mistral-tiny, mistral-small, mistral-medium")
    print("Введите 'exit' для выхода")
    print("Введите 'model название_модели' для смены модели")
    print("Введите 'reset_prompt' для смены системного промта")
    print("Введите 'show_prompt' для показа текущего системного промта")
    print("Введите 'show_history' для показа истории диалога")
    print("Введите 'clear_history' для очистки истории диалога")
    print("=" * 50)
    
    current_model = "mistral-tiny"
    chat_history = []
    
    while True:
        try:
            question = input("\nВы: ").strip()
            
            if question.lower() == "exit":
                print("Завершение программы. До свидания!")
                break
            elif question.lower().startswith("model "):
                new_model = question[6:].strip()
                available_models = ["mistral-tiny", "mistral-small", "mistral-medium"]
                if new_model in available_models:
                    current_model = new_model
                    print(f"✅ Модель изменена на: {current_model}")
                else:
                    print(f"❌ Неизвестная модель. Доступные: {', '.join(available_models)}")
                continue
            elif question.lower() == "reset_prompt":
                system_prompt = select_system_prompt()
                chat_history = []
                continue
            elif question.lower() == "show_prompt":
                if system_prompt:
                    print(f"Текущий системный промт: {system_prompt}")
                else:
                    print("Системный промт не установлен")
                continue
            elif question.lower() == "show_history":
                show_history(chat_history)
                continue
            elif question.lower() == "clear_history":
                chat_history = []
                print("✅ История диалога очищена")
                continue
            elif not question:
                continue
                
            print("Mistral: ", end="", flush=True)
            
            answer = get_response(question, client, current_model, system_prompt, chat_history)
            print(answer)
            
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            
            chat_history = limit_history(chat_history, 6)
            
        except KeyboardInterrupt:
            print("\n\nЗавершение программы. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
