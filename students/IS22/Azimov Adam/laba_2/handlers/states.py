from aiogram.fsm.state import State, StatesGroup


class UserState(StatesGroup):
    waiting = State()
    name = State()
    sure = State()


class DialogState(StatesGroup):
    active = State()
