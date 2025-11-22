# from dotenv import load_dotenv
import os
from dotenv import load_dotenv
load_dotenv()
gpt_api = os.getenv("GPT_API")
bot_api = os.getenv("BOT_API")
model = "gpt-4o-mini"
database_url = os.getenv("DATABASE_URL")
