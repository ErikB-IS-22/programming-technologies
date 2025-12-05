from aiogram import F
from utils.loader import dp
import logging
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, ContentType
from utils.gpt import get_response, client
import json
from keyboard import keyboard_choose

from db import get_session, Prompt, Dialog   

waiting_for_prompt = {}


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message):
    await message.answer("Вы отправили картинку!")


@dp.message(lambda message: message.text == "Выбрать промт")
async def get_choose_promt(message: Message):
    user_id = str(message.from_user.id)

    with get_session() as session:
        prompts = (
            session.query(Prompt)
            .filter_by(user_id=user_id)
            .order_by(Prompt.id.desc())
            .all()
        )

    if not prompts:
        await message.answer("У вас пока нет сохранённых промтов")
        return

    keyboard = ReplyKeyboardMarkup(keyboard=[], resize_keyboard=True)

    for p in prompts:
        short = p.text[:30] + "..." if len(p.text) > 30 else p.text
        btn = KeyboardButton(text=f"Промт №{p.id}){short}")
        keyboard.keyboard.append([btn])

    keyboard.keyboard.append([KeyboardButton(text="Отмена")])
    await message.answer("Выберите промт:", reply_markup=keyboard)


@dp.message(lambda message: message.text == "Отмена")
async def back(message: Message):
    await message.answer("Действие отменено", reply_markup=keyboard_choose)


@dp.message(lambda message: message.text.startswith("Промт №"))
async def choose_promt(message: Message):
    text = message.text
    part = text.split("№")[1]
    prompt_id_str = part.split(")")[0]
    prompt_id = int(prompt_id_str)
    
    with get_session() as session:
        prompt_obj = session.query(Prompt).filter_by(id=prompt_id).first()

        if not prompt_obj:
            await message.answer("Промт не найден")
            return

        full_prompt = prompt_obj.text
        user_id = str(message.from_user.id)

        dialog = (
            session.query(Dialog)
            .filter_by(user_id=user_id)
            .order_by(Dialog.id.asc())
            .first()
        )

        if not dialog:
            await message.answer("Диалог не найден")
            return

        obj_json = json.dumps({
            "role": "system",
            "content": full_prompt
        }, ensure_ascii=False)

        dialog.data = obj_json
        session.commit()

    await message.answer("Промт выбран и сохранён", reply_markup=ReplyKeyboardRemove())


@dp.message(lambda message: message.text == "Добавить промт")
async def handle_promt(message: Message):
    user_id = message.from_user.id
    waiting_for_prompt[user_id] = True
    await message.answer("Напиши свой промт")


@dp.message(lambda message: message.from_user.id in waiting_for_prompt)
async def handle_save_promt(message: Message):
    user_id = str(message.from_user.id)
    prompt_text = message.text

    with get_session() as session:
        prompt = Prompt(user_id=user_id, text=prompt_text)
        session.add(prompt)
        session.commit()

    del waiting_for_prompt[int(user_id)]
    await message.answer(f"Промт сохранен: {prompt_text}")


@dp.message()
async def message_handler(message: Message):
    user_id = str(message.from_user.id)
    user_text = message.text

    try:
        with get_session() as session:
            dialog = (
                session.query(Dialog)
                .filter_by(user_id=user_id)
                .order_by(Dialog.id.asc())
                .first()
            )

            if not dialog:
                dialog = Dialog(user_id=user_id, data="[]")
                session.add(dialog)
                session.commit()

            obj_data = dialog.data

            try:
                messages = json.loads(obj_data) if obj_data else []
                if isinstance(messages, dict):
                    messages = [messages]
            except:
                messages = []

            messages.append({"role": "user", "content": user_text})

            assistant_text = await get_response(messages, client)

            messages.append({"role": "assistant", "content": assistant_text})

            dialog.data = json.dumps(messages, ensure_ascii=False)
            print(dialog.data)
            session.commit()

        await message.answer(assistant_text)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        await message.answer("Произошла ошибка при получении ответа")
