import sqlite3

def create_tables(cursor):
    cursor.execute("""CREATE TABLE promt
                (id INTEGER PRIMARY KEY AUTOINCREMENT,  
                text TEXT)
            """)