import json
from aiogram.fsm.context import FSMContext
from aiogram.filters import StateFilter
from aiogram.types import Message
from aiogram import F

from handlers.states import DialogState, UserState
from handlers.keyboard import inline_sure, end_dialog, start_dialog, my_dialogs, inline_dialogs, reply_active_dialog, reply_not_dialog
from database.dependency import AsyncSessionLocal
from database.dao import DialogsDAO, UserDAO
from database.models import Dialogs
from utils.gpt import get_response, client
from utils.prompt import get_dialog_prompt
from utils.loader import dp


# ПРИ UserState ---------------------------------------------------------------------------)
@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("Вы отправили картинку!")


@dp.message(F.text, StateFilter(None))
async def user_init(message: Message, state: FSMContext):
    await state.set_state(UserState.waiting)
    await state.update_data(telegram_id=str(message.from_user.id))

    async with AsyncSessionLocal() as session:
        user = await UserDAO.find_one_or_none(db=session, telegram_id=str(message.from_user.id))
        if not user:
            user = await UserDAO.add(
                db=session,
                username=message.from_user.username,
                telegram_id=str(message.from_user.id),
                full_name=message.from_user.full_name)

    await message.answer(f"Используй клавиатуру, чтобы начать диалог", reply_markup=reply_not_dialog())


@dp.message(lambda message: message.text == start_dialog, StateFilter(UserState.waiting))
async def user_init(message: Message, state: FSMContext):
    async with AsyncSessionLocal() as session:
        user = await UserDAO.find_one_or_none(db=session, telegram_id=str(message.from_user.id))

        dialogs = await DialogsDAO.find_all(db=session, user_id=user.id, is_active=True)

        if dialogs:
            update_data = [{"id": dialog.id, "is_active": False}
                           for dialog in dialogs]
            await DialogsDAO.update_many(db=session, updates=update_data)

    await message.answer(f"Напишите, как мне к тебе обращаться")
    await state.set_state(UserState.name)


@dp.message(UserState.name)
async def activate_testing(message: Message, state: FSMContext):
    name = message.text
    await message.answer(f"Вы уверены, в имени?: {name}", reply_markup=inline_sure())
    await state.update_data(name=message.text)

    await state.set_state(UserState.sure)


@dp.message(UserState.sure)
async def activate_testing(message: Message, state: FSMContext):
    data: dict = await state.get_data()
    name = data.get('name')
    await message.answer(f"Вы уверены, в имени?: {name}", reply_markup=inline_sure())


@dp.message(lambda message: message.text == my_dialogs, StateFilter(UserState.waiting))
async def user_init(message: Message, state: FSMContext):
    async with AsyncSessionLocal() as session:
        user = await UserDAO.find_one_or_none(db=session, telegram_id=str(message.from_user.id))
        dialogs = await DialogsDAO.find_all(db=session, user_id=user.id)

    await message.answer(f"У вас всего {len(dialogs)} диалогов", reply_markup=inline_dialogs(dialogs))

# ПРИ DialogState ---------------------------------------------------------------------------)


@dp.message(lambda message: message.text == end_dialog, StateFilter(DialogState.active))
async def create_bot(message: Message, state: FSMContext):
    await state.clear()

    async with AsyncSessionLocal() as session:
        user = await UserDAO.find_one_or_none(db=session, telegram_id=str(message.from_user.id))
        dialog = await DialogsDAO.find_one_or_none(db=session, user_id=user.id, is_active=True)

        await DialogsDAO.update(db=session, model_id=dialog.id, is_active=False)
    await state.set_state(UserState.waiting)
    await state.update_data(telegram_id=str(message.from_user.id))
    await message.answer(f"Вы завершили диалог", reply_markup=reply_not_dialog())


@dp.message(DialogState.active)
async def activate_testing(message: Message, state: FSMContext):
    user_message = message.text
    async with AsyncSessionLocal() as session:
        user = await UserDAO.find_one_or_none(db=session, telegram_id=str(message.from_user.id))
        dialog: Dialogs = await DialogsDAO.find_one_or_none(db=session, user_id=user.id, is_active=True)
        dialog.list_messages.append({"role": "user", "content": user_message})
        prompt = get_dialog_prompt(
            dialog.list_messages, username=dialog.username)
        answer = await get_response(prompt, client)
        assistant_answer = json.loads(
            "{"+answer[answer.find("{")+1:answer.rfind("}")]+"}")

        dialog.list_messages.append(assistant_answer)
        await DialogsDAO.update(db=session, model_id=dialog.id, list_messages=dialog.list_messages)
    await message.answer(assistant_answer.get("content"), reply_markup=reply_active_dialog())
