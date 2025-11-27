from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from config import token

dp = Dispatcher()
bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))