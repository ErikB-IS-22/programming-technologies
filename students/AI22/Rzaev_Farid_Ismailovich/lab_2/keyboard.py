from aiogram.types import Message,ReplyKeyboardMarkup, KeyboardButton

keyboard_choose = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Добавить промт"), KeyboardButton(text="Выбрать промт")],
    ],
    resize_keyboard=True, 
)