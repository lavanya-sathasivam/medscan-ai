from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DATABASE_PATH = Path(__file__).resolve().parent / "database.db"


def _get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                patient_name TEXT,
                patient_age INTEGER,
                image_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                triage TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(cursor, "patient_id", "TEXT")
        _ensure_column(cursor, "patient_name", "TEXT")
        _ensure_column(cursor, "patient_age", "INTEGER")
        connection.commit()


def _ensure_column(cursor: sqlite3.Cursor, column_name: str, column_type: str) -> None:
    columns = {row["name"] for row in cursor.execute("PRAGMA table_info(scans)").fetchall()}
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE scans ADD COLUMN {column_name} {column_type}")


def save_scan(
    patient_id: str,
    patient_name: str,
    patient_age: int,
    image_name: str,
    prediction: str,
    confidence: float,
    triage: str,
) -> None:
    with _get_connection() as connection:
        connection.execute(
            """
            INSERT INTO scans (
                patient_id,
                patient_name,
                patient_age,
                image_name,
                prediction,
                confidence,
                triage
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_id, patient_name, patient_age, image_name, prediction, confidence, triage),
        )
        connection.commit()


def fetch_scans(patient_id: str | None = None) -> pd.DataFrame:
    with _get_connection() as connection:
        query = """
            SELECT patient_id, patient_name, patient_age, image_name, prediction, confidence, triage, timestamp
            FROM scans
        """
        parameters: tuple[str, ...] = ()
        if patient_id:
            query += " WHERE patient_id = ?"
            parameters = (patient_id,)
        query += " ORDER BY datetime(timestamp) DESC, id DESC"
        rows = connection.execute(query, parameters).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "patient_name",
                "patient_age",
                "image_name",
                "prediction",
                "confidence",
                "triage",
                "timestamp",
            ]
        )

    return pd.DataFrame([dict(row) for row in rows])
