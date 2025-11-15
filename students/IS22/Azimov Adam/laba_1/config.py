# from dotenv import load_dotenv
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")
prompt = os.getenv("PROMPT")
temperature = os.getenv("TEMPERATURE")
model = "gpt-4o-mini"
