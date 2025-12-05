import json
import logging
from utils.loader import dp
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from keyboard import keyboard_choose
from db import get_session, Dialog  


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        user_id = str(message.from_user.id)

        with get_session() as session:
            dialog = (
                session.query(Dialog)
                .filter_by(user_id=user_id)
                .order_by(Dialog.id.asc())
                .first()
            )

            if not dialog:
                dialog = Dialog(
                    user_id=user_id,
                    data="[]"  
                )
                session.add(dialog)
                session.commit()

        await message.answer(
            f"Привет, {message.from_user.full_name}, я твой бот-ассистент! "
            "Можешь задавать мне вопросы, и я буду отвечать на них. "
            "Пожалуйста, помни про свой баланс на счету аккаунта в OpenAI и не ддось меня без необходимости)"
        )
        await message.answer("Выберите действие:", reply_markup=keyboard_choose)

    except Exception as e:
        logging.error(f"Error occurred: {e}")



@dp.message(Command("reset-context"))
async def reset_context(message: Message):
    user_id = str(message.from_user.id)
    try:
        with get_session() as session:
            dialog = (
                session.query(Dialog)
                .filter_by(user_id=user_id)
                .order_by(Dialog.id.asc())
                .first()
            )

            if not dialog:
                await message.answer("Диалог не найден")
                return

            dialog.data = json.dumps([], ensure_ascii=False)
            session.commit()

        await message.answer(
            "Контекст диалога успешно сброшен!",
            reply_markup=keyboard_choose
        )

    except Exception as e:
        logging.error(f"Error occurred while resetting context: {e}")
        await message.answer("Произошла ошибка при сбросе контекста")
