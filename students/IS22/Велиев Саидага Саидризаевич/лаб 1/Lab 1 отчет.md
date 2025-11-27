# Лабораторная работа №1: Знакомство с OpenAI API. Написание простого текстового ассистента

## Цель работы

Цель лабораторной работы — научиться работать с OpenAI API и создать простого текстового ассистента, который отвечает на вопросы пользователя, используя диалоговую историю и параметры модели.

## Инструменты и настройки

- **Язык программирования**: Python.
- **Библиотеки**:
  - `openai` — для работы с API.
  - `dotenv` — для загрузки переменных окружения (например, API-ключ).
  

## Реализованные задачи

В коде были реализованы следующие задачи:

1. **Использование системного промпта через переменную окружения `.env`**:
    
    ```
    system_prompt = os.getenv("SYSTEM_PROMPT")

    client = OpenAI(api_key=api_key)

    def get_response(system_prompt: str, user_text: str, client: OpenAI):
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        return response.output_text
    ```

    Переменная system_prompt извлекается из файла .env с помощью os.getenv("SYSTEM_PROMPT").

    Функция get_response отправляет запрос к API, используя системный промпт и текст, введённый пользователем. Ответ от модели возвращается и выводится на экран.
    **Результат работы:**

    ![Системный промт](./Screenshots/Системный%20промт.png)


2. **Работа с параметром `temperature`**:
    ```
    system_prompt = os.getenv("SYSTEM_PROMPT")
    temperature = os.getenv("TEMPERATURE")


    client = OpenAI(api_key=api_key)

    def get_response(system_prompt: str, user_text: str, temperature: str client: OpenAI):
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=float(temperature)
        )
        return response.output_text
    ```
   В процессе работы с языковой моделью был реализован эксперимент с параметром `temperature`, который управляет случайностью выводимых ответов. 
   На gpt-5-nano не поддерживается изменение этого параметра, пришлось сменить на gpt=4.1-nano.
   Я использовал значение температуры 0.1 для получения более предсказуемых и стабильных ответов 
   
   ![Системный промт](./Screenshots/Temprature%200.1.png) однако результат оказался схож с температурой 1.0 (дефолтное значение) 
   
   ![Системный промт](./Screenshots/Temprature%201.png) При температуре 2.0 модель начала генерировать более случайные и менее логичные ответы, иногда теряя тему и выдавая бессмысленные фразы на разных языках 
   
   ![Системный промт](./Screenshots/Temprature%202.png)
3. **Ведение истории диалога (контекста переписки)**:
    ```
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
    ```
   Для того, чтобы ИИ помнил контекст общения с пользователем, была реализована система ведения истории диалога. Контекст переписки ограничивался 6 последними сообщениями (3 от пользователя и 3 от ИИ). Это позволяло модели помнить предыдущие вопросы и ответы, улучшая качество взаимодействия и позволяя более точно реагировать на новые запросы пользователя
   
   ![Системный промт](./Screenshots/История%20сообщений.png)

4. **Итоговый код**
```
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
```



