import json
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

class DialogueHistory:
    def __init__(self, max_messages: int = 6, filename: str = "dialogue_history.json"):
        self.max_messages = max_messages
        self.filename = filename
        self.history: List[Dict] = []
        self.load_history()
    
    def add_message(self, role: str, content: str):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(message)
        
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]
        
        self.save_history()
    
    def get_messages(self) -> List[Dict]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.history]
    
    def clear(self):
        self.history.clear()
        self.save_history()
    
    def save_history(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "last_updated": datetime.now().isoformat(),
                        "total_messages": len(self.history)
                    },
                    "messages": self.history
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def load_history(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get("messages", [])
        except Exception:
            self.history = []

def get_response(messages: List[Dict], client: OpenAI):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages
    )
    return response

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    system_prompt = os.getenv("SYSTEM_PROMPT")
    temperature = float(os.getenv("TEMPERATURE"))
    
    client = OpenAI(api_key=api_key)
    dialogue_history = DialogueHistory(max_messages=6)
    
    print("Введите ваш вопрос (или 'exit' для выхода):")
    
    while True:
        question = input("Вы: ")
        
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        elif question.lower() == "clear":
            dialogue_history.clear()
            print("История диалога очищена.")
            continue
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(dialogue_history.get_messages())
        messages.append({"role": "user", "content": question})
        
        try:
            answer = get_response(messages, client)
            print("AI:", answer.output_text)
            
            dialogue_history.add_message("user", question)
            dialogue_history.add_message("assistant", answer.output_text)
            
        except Exception as e:
            print("Ошибка:", e)
            print("Ждем 10 секунд перед повторной попыткой...")
            time.sleep(10)

if __name__ == "__main__":
    main()