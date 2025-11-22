import os
import requests
from dotenv import load_dotenv

load_dotenv()

system_prompt = os.getenv("SYSTEM_PROMPT")
temp = 0.5
history = []

def creat_prompt(system_prompt, history, user_input):
    prompt = system_prompt + "\n\nИстория диалога:\n"

    for m in history:
        if m["role"] == "user":
            prompt += f"Пользователь: {m['content']}\n"
        else:
            prompt += f"Ассистент: {m['content']}\n"

    prompt += f"\nПользователь: {user_input}\nАссистент:"

    return prompt
def get_response(text: str):

    full_prompt = creat_prompt(system_prompt, history, text)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:4b",
            "prompt": full_prompt,
            "temperature": temp,
            "stream": False
        }
    )

    return response.json()['response']

if __name__ == "__main__":

    print("Системный промпт загружен из .env:")
    print(system_prompt)
    print(f"\nПараметр temperature в модели: {temp}")
    print("\nДобавлена функция сохранения диалога до 6 последних сообщений.")
    print("\nВведите ваш вопрос (или 'exit' для выхода):")

    while True:
        question = input("Вы: ")
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        
        answer = get_response(question)
        print("AI:", answer)
        
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        history = history[-6:]