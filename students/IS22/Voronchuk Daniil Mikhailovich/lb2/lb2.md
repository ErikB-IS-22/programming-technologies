# Лабораторная работа №2. Лабораторная работа №2. Простейший чат-бот в Telegram

## План

1. Настройка окружения;
2. Написание основных функций бота;
3. Задания.

## Процесс выполнения

- Выполнив все действия, указанные в методичке ко второй лабораторной работе, я приступил к реализации заданий

### 1 Добавление к ассистенту системного промпта

- В файле config.py добавлена переменная SYSTEM_PROMPT:

```
SYSTEM_PROMPT = "Ты Учитель для детей младших классов и все объясняешь понятно"
```

В функции get_response (файл utils/gpt.py) системный промпт используется при формировании запроса к модели:

```python
from config import SYSTEM_PROMPT

async def get_response(message: str, client: AsyncOpenAI, user_id: int, user_name: str = "") -> str:
    try:
        prompt = f"{SYSTEM_PROMPT}\nПользователь: {user_name}\nВопрос: {message}"
        response = await client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"

```

- Пример представлен на рисунке ниже

![screen](Screenshots/1.jpg)

### 2 Реализация функционала персонализированного обращения бота к пользователю по имени

- В функции обработки сообщений (handlers/messages.py) получаем имя пользователя из объекта Message

```python

user_name = message.from_user.full_name

```

- Имя пользователя передаётся в функцию get_response для формирования запроса к модели:

```python

response = await get_response(message.text, client, message.from_user.id, user_name)

```

- В utils/gpt.py системный промпт и имя пользователя используются при формировании запроса к модели

```python

prompt = f"{SYSTEM_PROMPT}\nПользователь: {user_name}\nВопрос: {message}"

```

Пример обращения представлен на рисунке ниже

![screen](Screenshots/2.jpg)
