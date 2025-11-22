import sqlite3
from openai import OpenAI
import httpx
from dotenv import load_dotenv
import os 
from db import create_tables

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
proxy = os.getenv("PROXY")
http_client = httpx.Client(proxy=proxy)
client = OpenAI(
    api_key=api_key,
    http_client=http_client  
)



def get_response(text, client: OpenAI, temperature=1):
    response = client.responses.create(model="gpt-4o-mini", input=text,temperature=temperature)
    return response
sytem_promt = ""
history = []
hh=0
if __name__ == "__main__":
    con = sqlite3.connect("ai.db")
    cursor = con.cursor()
    #create_tables(cursor=cursor)
   
    
    print("Выберети \n 1.использовать системный промт \n 2.без промта ")
    cmd = input()
    if(cmd == "1"):
        isTaket = False
        while(not isTaket):
            print("Выберети \n 1.создать промт \n 2.выбрать промт ")
            cmd = input()
            if(cmd == "1"):
                text=input()
                cursor.execute("INSERT INTO promt (text) VALUES (?)",(text,))
                con.commit()
                pass
            elif(cmd == "2"):
                cursor.execute("SELECT * FROM promt")
                pomt_db = cursor.fetchall()
                if(len(pomt_db) == 0):
                    print("Нет промтов создай")
                else:
                    print("Выберети промт ")
                    ii = 1
                    for p in pomt_db:
                        print(ii," ",p[1])
                        ii+=1
                    num=int(input())
                    sytem_promt = pomt_db[num-1][1]
                    isTaket = True
                pass
    if(sytem_promt):
        history.append({"role": "system", "content": sytem_promt})
    else:
        history = []
    print("Введите ваш вопрос (или 'exit' для выхода):")
    while True:
        question = input("Вы: ")    
        hh+=2
        if question.lower() == "exit":
            print("Завершение программы.")
            break
        history.append({"role": "user", "content": question})
        answer = get_response(history, client)
       
        history.append({"role": "assistant", "content": answer.output_text})
        if(hh==6):
            #print(history)
            print(sytem_promt)
            hh=0
            history = []
            if(sytem_promt):
                history=[{"role": "system", "content": sytem_promt}]

        print("AI:", answer.output_text)