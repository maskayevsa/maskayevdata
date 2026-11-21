import os
import sqlite3
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "events.db")


def get_connection(db_path=DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")  # Безопасная параллельная запись и чтение (CV + Web)
    return conn


def init_db(db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()

    # 1. Справочник известных людей
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            note TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    # 2. Векторные слепки лиц (эмбеддинги) для дообучения
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            source_snapshot TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
        )
    """)

    # 3. Журнал событий (включая события песочницы)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            person_id INTEGER,
            confidence REAL DEFAULT 0.0,
            direction TEXT DEFAULT 'unknown',
            snapshot_path TEXT NOT NULL,
            video_file TEXT,
            video_offset_sec REAL DEFAULT 0.0,
            source_type TEXT DEFAULT 'realtime',
            is_sandbox INTEGER DEFAULT 0,
            raw_embedding BLOB,
            FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE SET NULL
        )
    """)

    # 4. Реестр интервалов работы онлайн-распознавания (для функции "Проверить архив")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyzed_intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL,
            mode TEXT DEFAULT 'realtime'
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("База данных успешно инициализирована:", DB_PATH)
