import os
import json
import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()

def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            is_active INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def save_system_prompt(name, content):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('UPDATE system_prompts SET is_active = 0')
    
    cursor.execute('''
        INSERT INTO system_prompts (name, content, is_active)
        VALUES (?, ?, 1)
    ''', (name, content))
    
    conn.commit()
    conn.close()

def get_active_system_prompt():
    env_prompt = os.getenv('DEEPSEEK_SYSTEM_PROMPT')
    if env_prompt:
        return env_prompt
    
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT content FROM system_prompts WHERE is_active = 1')
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else "You are a helpful assistant"

def save_message(role, content):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (role, content)
        VALUES (?, ?)
    ''', (role, content))
    
    conn.commit()
    conn.close()

def get_recent_history():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content FROM chat_history 
        ORDER BY timestamp DESC 
        LIMIT 6
    ''')
    
    messages = cursor.fetchall()
    conn.close()
    
    history = [{"role": role, "content": content} for role, content in reversed(messages)]
    return history

def clear_history():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM chat_history')
    
    conn.commit()
    conn.close()

def manage_system_prompts():
    while True:
        print("\n=== Управление системными промптами ===")
        print("1. Просмотреть сохраненные промпты")
        print("2. Добавить новый промпт")
        print("3. Активировать промпт")
        print("4. Вернуться в главное меню")
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
            view_prompts()
        elif choice == "2":
            add_prompt()
        elif choice == "3":
            activate_prompt()
        elif choice == "4":
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

def view_prompts():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, content, is_active FROM system_prompts ORDER BY id')
    prompts = cursor.fetchall()
    
    conn.close()
    
    if not prompts:
        print("Нет сохраненных промптов.")
        return
    
    print("\nСохраненные промпты:")
    for prompt_id, name, content, is_active in prompts:
        status = "АКТИВЕН" if is_active else "неактивен"
        print(f"{prompt_id}. {name} [{status}]")
        print(f"   {content[:100]}..." if len(content) > 100 else f"   {content}")
        print()

def add_prompt():
    print("\nДобавление нового системного промпта:")
    name = input("Введите название промпта: ")
    print("Введите содержание промпта (введите 'END' на новой строке для завершения):")
    
    content_lines = []
    while True:
        line = input()
        if line.strip() == 'END':
            break
        content_lines.append(line)
    
    content = '\n'.join(content_lines)
    
    save_system_prompt(name, content)
    print(f"Промпт '{name}' сохранен и активирован!")

def activate_prompt():
    view_prompts()
    
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM system_prompts')
    prompts = cursor.fetchall()
    
    if not prompts:
        print("Нет доступных промптов для активации.")
        conn.close()
        return
    
    try:
        prompt_id = int(input("Введите ID промпта для активации: "))
        
        cursor.execute('UPDATE system_prompts SET is_active = 0')
        cursor.execute('UPDATE system_prompts SET is_active = 1 WHERE id = ?', (prompt_id,))
        
        if cursor.rowcount > 0:
            print("Промпт активирован!")
        else:
            print("Промпт с таким ID не найден.")
        
        conn.commit()
    except ValueError:
        print("Неверный формат ID.")
    
    conn.close()

def get_response(user_input):
    url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
    
    system_prompt = get_active_system_prompt()
    history = get_recent_history()
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": messages,
        "temperature": 0.5
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    if 'choices' in data and len(data['choices']) > 0:
        text = data['choices'][0]['message']['content']
        
        save_message("user", user_input)
        
        if '</think>' in text:
            assistant_response = text.split('</think>')[1].strip()
        else:
            assistant_response = text.strip()
        
        save_message("assistant", assistant_response)
        
        return assistant_response
    else:
        return "Ошибка: не удалось получить ответ от API"

if __name__ == "__main__":
    init_db()
    
    print("=== Чат с DeepSeek-R1 ===")
    
    while True:
        print("\nГлавное меню:")
        print("1. Начать общение")
        print("2. Управление системными промптами")
        print("3. Очистить историю диалога")
        print("4. Выйти")
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
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
        elif choice == "2":
            manage_system_prompts()
        elif choice == "3":
            clear_history()
            print("История диалога очищена!")
        elif choice == "4":
            print("Завершение программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
