from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from config import OPENAI_API_KEY, TOKEN
from openai import AsyncOpenAI

dp = Dispatcher()
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
client = AsyncOpenAI(api_key=OPENAI_API_KEY)