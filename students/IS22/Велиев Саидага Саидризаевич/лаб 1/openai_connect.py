import sys
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv('data.env')

api_key = os.getenv("OPENAI_API_KEY")
prompt = os.getenv("SYSTEM_PROMPT")
temperature = os.getenv("TEMPERATURE")

client = OpenAI(api_key=api_key)

dialog_history = []

def get_response(text: str, dialog_history: list, client: OpenAI):
    dialog_history.append({"role": "user", "content": text})

    if len(dialog_history) > 6:
        dialog_history.pop(0)

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=dialog_history,
        temperature=float(temperature)
    )

    ai_message = response.output_text
    dialog_history.append({"role": "assistant", "content": ai_message})

    return ai_message

if __name__ == "__main__":
    print("Введите ваш вопрос (или 'exit' для выхода):")
    while True:
        question = input("Вы: ")
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        answer = get_response(question, dialog_history, client)
        print("AI:", answer)
