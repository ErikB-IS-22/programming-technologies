import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def init_db():
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prompt(prompt: str):
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO prompts (content) VALUES (?)", (prompt,))
    conn.commit()
    conn.close()

def get_last_prompt():
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM prompts ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def choose_system_prompt():
    env_prompt = os.getenv("SYSTEM_PROMPT")
    db_prompt = get_last_prompt()

    print("\nВыберите источник системного промпта:")
    print("1 — Из .env")
    print("2 — Из базы данных (последний сохранённый)")
    print("3 — Ввести вручную")

    choice = input("Ваш выбор: ").strip()

    if choice == "1" and env_prompt:
        print(" Используется промпт из .env")
        return env_prompt
    elif choice == "2" and db_prompt:
        print(" Используется промпт из базы данных")
        return db_prompt
    elif choice == "3":
        prompt = input("Введите системный промпт: ")
        save_prompt(prompt)
        print(" Промпт сохранён в базу данных")
        return prompt
    else:
        print(" Неверный выбор — используется стандартный промпт.")
        return "Ты полезный и доброжелательный ассистент."


def update_history(history, role, content):
    history.append({"role": role, "content": content})
    if len(history) > 7:  
        history = [history[0]] + history[-6:]  
    return history

def get_response(history, temperature):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=temperature
    )
    return response.choices[0].message.content

def main():
    init_db()
    system_prompt = choose_system_prompt()
    temperature = float(input("Введите значение temperature (0.0–1.5): ") or 0.7)

    history = [{"role": "system", "content": system_prompt}]
    print("\n=== Начало диалога ===")
    print("(Введите 'exit' для выхода)\n")

    while True:
        user_input = input("Вы: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Завершение программы.")
            break

        history = update_history(history, "user", user_input)
        answer = get_response(history, temperature)
        print("AI:", answer)
        history = update_history(history, "assistant", answer)

if __name__ == "__main__":
    main()
