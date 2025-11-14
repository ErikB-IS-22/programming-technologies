import time
import sys
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

ai_api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=ai_api_key)

def get_response(text: str, client: OpenAI):
    response = client.responses.create(model="gpt-4.1-nano", input=text)
    return response

if __name__ == "__main__":
    print("Введите ваш вопрос (или 'exit' для выхода):")
    while True:
        question = input("Вы: ")
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        
        try:
            answer = get_response(question, client)
            print("AI:", answer.output_text)
        except Exception as e:
            print("Ошибка:", e)
            print("Ждем 10 секунд перед повторной попыткой...")
            time.sleep(10)