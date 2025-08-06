import sqlite3
from datetime import datetime

DB_NAME = "guess_record.db"

def init_db():
    conn=sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              folder_name TEXT,
              real_class TEXT,
              model_no TEXT,
              clothes_type TEXT

              )
           """)
    
    conn.commit()
    conn.close()

def save_prediction(folder_name, real_class, model_no, clothes_type):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (folder_name, real_class, model_no, clothes_type)
        VALUES (?, ?, ?, ?)
    """, (folder_name, real_class, model_no, clothes_type))
    conn.commit()
    conn.close()

