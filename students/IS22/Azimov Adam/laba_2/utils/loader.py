from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from config import bot_api

dp = Dispatcher()
bot = Bot(token=bot_api, default=DefaultBotProperties(
    parse_mode=ParseMode.HTML))
