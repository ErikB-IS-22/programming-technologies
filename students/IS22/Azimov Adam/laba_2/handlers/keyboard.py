from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from database.models import Dialogs

start_dialog = "Начать диалог"
end_dialog = "Завершить диалог"
my_dialogs = "Мои диалоги"

text_btn_yes_sure = "✅ Да, верно"
text_btn_not_sure = "❌ Нет"
btn_yes_sure_callback = "btn_yes_sure"
btn_not_sure_callback = "btn_not_sure"


def reply_not_dialog() -> ReplyKeyboardMarkup:
    list_buttons = [
        [
            KeyboardButton(text=start_dialog),
            KeyboardButton(text=my_dialogs)
        ],
    ]
    keyboard = ReplyKeyboardMarkup(
        keyboard=list_buttons,
        resize_keyboard=True)
    return keyboard


def reply_active_dialog() -> ReplyKeyboardMarkup:
    list_buttons = [
        [
            KeyboardButton(text=end_dialog)
        ]
    ]
    keyboard = ReplyKeyboardMarkup(
        keyboard=list_buttons,
        resize_keyboard=True)
    return keyboard


def inline_sure() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text=text_btn_yes_sure,
                              callback_data=btn_yes_sure_callback)  ,
         InlineKeyboardButton(text=text_btn_not_sure,
                              callback_data=btn_not_sure_callback)],
    ]

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=buttons
    )
    return keyboard


def inline_dialogs(dialogs: list[Dialogs]) -> InlineKeyboardMarkup:
    buttons = []
    line_buttons = []
    count_dialogs = len(dialogs)
    count_buttons_in_line = 2 if count_dialogs % 2 == 0 else 3
    print(count_dialogs)
    print(count_buttons_in_line)
    for index in range(1, count_dialogs+1):
        username = dialogs[index-1].username
        create_at = dialogs[index-1].created_at.strftime("%d.%m.%Y %H:%M")
        if index % count_buttons_in_line == 0:
            line_buttons.append(InlineKeyboardButton(text=f"№{index} {username} {create_at}",
                                                     callback_data=f"dialog_{index}"))
            buttons.append(line_buttons)
            line_buttons = []
        else:
            line_buttons.append(InlineKeyboardButton(text=f"№{index} {username} {create_at}",
                                                     callback_data=f"dialog_{index}"))
    if line_buttons:
        buttons.append(line_buttons)

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=buttons
    )
    return keyboard
