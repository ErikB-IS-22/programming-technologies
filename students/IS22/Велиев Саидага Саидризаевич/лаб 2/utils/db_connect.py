import sqlite3
import json

DB_PATH = "dialog_history.db"

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dialog_history (
        user_id TEXT PRIMARY KEY,
        history TEXT
    )
    ''')
    conn.commit()
    conn.close()

create_table()

def get_dialog_history(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT history FROM dialog_history WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if result:
        dialog_history = json.loads(result[0])
    else:
        dialog_history = []
    conn.close()
    return dialog_history

def save_dialog_history(user_id: str, dialog_history_actual: list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO dialog_history (user_id, history)
    VALUES (?, ?)
    """, (user_id, json.dumps(dialog_history_actual)))
    conn.commit()
    conn.close()
