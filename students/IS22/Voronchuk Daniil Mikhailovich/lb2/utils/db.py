import sqlite3

conn = sqlite3.connect("chat.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    user_id INTEGER,
    role TEXT,
    content TEXT
)
""")
conn.commit()

def save_message(user_id: int, role: str, content: str):
    cursor.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
                   (user_id, role, content))
    conn.commit()

def get_user_messages(user_id: int):
    cursor.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY rowid", (user_id,))
    return cursor.fetchall()

def reset_user_context(user_id: int):
    cursor.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
    conn.commit()
