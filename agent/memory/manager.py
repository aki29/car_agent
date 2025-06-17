import sqlite3
from datetime import datetime
from pathlib import Path
print(Path(__file__).parent.parent)
DB_PATH = Path(__file__).parent.parent / "data" / "ctk_memory.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory_key TEXT,
            memory_value TEXT,
            updated_at TEXT,
            UNIQUE(user_id, memory_key)
        )
        """)
        conn.commit()

def append_chat(user_id: str, role: str, content: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.now().isoformat()),
        )
        conn.commit()

def load_chat_history(user_id: str, limit: int = 20):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return list(reversed(rows))

def save_memory(user_id: str, key: str, value: str):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO user_memory (user_id, memory_key, memory_value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, memory_key)
            DO UPDATE SET memory_value=excluded.memory_value, updated_at=excluded.updated_at
        """, (user_id, key, value, datetime.now().isoformat()))
        conn.commit()

def load_memory(user_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {k: v for k, v in rows}

def clear_memory(user_id: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
        conn.commit()