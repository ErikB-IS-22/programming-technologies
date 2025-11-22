import logging
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.types import Message
from aiogram.filters import StateFilter
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext

from handlers.states import DialogState, UserState
from handlers.keyboard import btn_yes_sure_callback, btn_not_sure_callback, reply_active_dialog
from database.dependency import AsyncSessionLocal
from database.models import Dialogs
from database.dao import DialogsDAO, UserDAO
from utils.loader import dp


@dp.message(CommandStart(), StateFilter(None))
async def command_start_handler(message: Message, state: FSMContext) -> None:
    try:
        async with AsyncSessionLocal() as session:
            user = UserDAO.find_one_or_none(
                db=session,
                telegram_id=message.from_user.id)
            if not user:
                user = await UserDAO.add(
                    db=session,
                    username=message.message.from_user.username,
                    telegram_id=message.from_user.id,
                    full_name=message.from_user.full_name)
        await message.answer(f"""Привет, {message.from_user.full_name}, я твой бот-ассистент! 
Можешь задавать мне вопросы, и я буду отвечать на них. Пожалуйста, 
помни про свой баланс на счету аккаунта в OpenAI и не ддось меня без необходимости)""")
        await message.answer(f"Напиши как мне к тебе обращаться")
        await state.set_state(UserState.first_name)

    except Exception as e:
        logging.error(f"Error occurred: {e}")


@dp.callback_query(UserState.sure)
async def handle_callback_interaction(query: CallbackQuery, state: FSMContext):
    await query.answer()
    data: dict = await state.get_data()
    name = data.get('name')
    telegram_id = data.get('telegram_id')

    if query.data == btn_yes_sure_callback:
        await query.message.answer(f"Отлично, <i>{name}</i>! Теперь можем начать диалог.", reply_markup=reply_active_dialog())
        async with AsyncSessionLocal() as session:
            user = await UserDAO.find_one_or_none(db=session, telegram_id=telegram_id)
            await DialogsDAO.add(db=session, username=name, user_id=user.id)
        await state.clear()
        await state.set_state(DialogState.active)

    elif query.data == btn_not_sure_callback:
        await state.set_state(UserState.name)
        await query.message.answer(f"Напишите, как мне к тебе обращаться")


@dp.callback_query(UserState.waiting)
async def handle_callback_interaction(query: CallbackQuery, state: FSMContext):
    await query.answer()
    data: dict = await state.get_data()
    telegram_id = data.get('telegram_id')
    dialog_index = int(query.data.replace("dialog_", ""))
    async with AsyncSessionLocal() as session:
        user = await UserDAO.find_one_or_none(db=session, telegram_id=telegram_id)
        print(user)
        print(telegram_id)
        dialogs = await DialogsDAO.find_all(db=session, user_id=user.id)
    dialog: Dialogs = dialogs[dialog_index-1]

    create_at = dialog.created_at.strftime("%d.%m.%Y %H:%M")
    update_at = dialog.created_at.strftime("%d.%m.%Y %H:%M")
    count_messages = len(dialog.list_messages)

    message_text = f"""
<b>💬 Диалог #{dialog_index}</b>

<b>Пользователь:</b> {dialog.username}
<b>Сообщений:</b> {count_messages} 
<b>Начало:</b> {create_at}
<b>Завершение:</b> {update_at}

<u>📝 История диалога:</u>
"""
    for i, msg in enumerate(dialog.list_messages, 1):
        role_name = dialog.username if msg.get(
            "role") == "user" else "Ассистент"
        content = msg.get("content", "").replace('\n', ' ')

        message_text += f"\n - <b>{role_name}:</b> {content}"

    await query.message.answer(message_text)
