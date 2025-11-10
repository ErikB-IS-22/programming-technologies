from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
token = os.getenv("BOT_TOKEN")
prompt = os.getenv("SYSTEM_PROMPT")
temperature = os.getenv("TEMPERATURE")