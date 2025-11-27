import os

from dotenv import load_dotenv
from openai import OpenAI

from history import clear_history, load_history, save_history

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
system_prompt = os.getenv("SYSTEM_PROMPT")

client = OpenAI(api_key=api_key)

dialog_history = []

MAX_HISTORY = 6
dialog_history = load_history(MAX_HISTORY)


def get_response(text: str, client: OpenAI):
    global dialog_history
    dialog_history.append({"role": "user", "content": text})

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=dialog_history,
        instructions=system_prompt,
        temperature=0.8,
    )

    dialog_history.append({"role": "assistant", "content": response.output_text})

    dialog_history[:] = dialog_history[-MAX_HISTORY:]

    save_history(dialog_history)

    return response


if __name__ == "__main__":
    print("Введите ваш вопрос (или 'exit' для выхода):")
    while True:
        question = input("Вы: ")
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        if question.lower() == "clear":
            dialog_history = []
            clear_history()
            print("История диалога очищена.")
            continue

        answer = get_response(question, client)
        print("AI:", answer.output_text)
