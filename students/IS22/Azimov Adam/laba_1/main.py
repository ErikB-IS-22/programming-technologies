import json
from openai import OpenAI
from config import api_key, prompt, model, temperature

client = OpenAI(
    api_key=api_key)

dialogs = []


def get_response(text: str):
    global dialogs
    dialogs.append({"role": "user", "content": text})
    if len(dialogs)+1 > 6:
        dialogs = dialogs[2:6]

    messages = [{"role": "system", "content": prompt}] + dialogs
    response = client.responses.create(
        model=model,
        input=messages,
        temperature=float(temperature)
    )
    answer = response.output_text
    dialogs.append({"role": "assistant", "content": answer})
    return response.output_text


if __name__ == "__main__":

    print("Введите ваш вопрос (или 'exit' для выхода):")
    dialog_start = False
    user_answer = ""

    while True:
        question = input(f"Вы: {user_answer} ")

        if question.lower() == "exit":
            print("Завершение программы.")
            with open("dialogs.json", "w", encoding="utf-8") as f:
                json.dump({"dialog": dialogs}, f, ensure_ascii=False, indent=4)
            break

        answer = get_response(user_answer+question)
        change_answer = "{" + answer.split("{")[1].split("}")[0]+"}"
        data = json.loads(change_answer)
        user_answer = data.get("user")
        gpt_answer = data.get("assistant")
        print("AI:", gpt_answer)
