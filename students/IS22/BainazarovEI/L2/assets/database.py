import sqlite3

def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_message(role, content):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (role, content)
        VALUES (?, ?)
    ''', (role, content))
    
    conn.commit()
    conn.close()

def get_recent_history():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content FROM chat_history 
        ORDER BY timestamp DESC 
        LIMIT 6
    ''')
    
    messages = cursor.fetchall()
    conn.close()
    
    history = [{"role": role, "content": content} for role, content in reversed(messages)]
    return history

def clear_history():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM chat_history')
    
    conn.commit()
    conn.close()