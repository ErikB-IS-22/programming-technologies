from dotenv import load_dotenv
import os 

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("OPENAI_API_KEY")
PROXY = os.getenv("PROXY")