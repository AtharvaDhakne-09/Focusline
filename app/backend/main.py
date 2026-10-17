from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
APP_ARTIFACT = BASE_DIR / "model_outputs" / "application_models.joblib"
RECOVERY_ARTIFACT = BASE_DIR / "model_outputs" / "production_models.joblib"
DB_PATH = BASE_DIR / "app" / "prediction_history.sqlite3"

if not APP_ARTIFACT.exists() or not RECOVERY_ARTIFACT.exists():
    raise RuntimeError("Train application and production models before starting the API.")

APP_BUNDLE = joblib.load(APP_ARTIFACT)
RECOVERY_BUNDLE = joblib.load(RECOVERY_ARTIFACT)
FEATURES = APP_BUNDLE["features"]

app = FastAPI(title="Personalized Cognitive Interruption API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Context(BaseModel):
    sleep_hours: float = Field(7, ge=0, le=24)
    sleep_quality_1_5: float = Field(3, ge=1, le=5)
    pre_fatigue_1_5: float = Field(3, ge=1, le=5)
    pre_stress_1_5: float = Field(3, ge=1, le=5)
    pre_focus_1_5: float = Field(4, ge=1, le=5)
    task_familiarity_1_5: float = Field(4, ge=1, le=5)
    expected_difficulty_1_5: float = Field(3, ge=1, le=5)
    baseline_typing_wpm: float = Field(65, gt=0, le=300)
    baseline_error_rate: float = Field(0.03, ge=0, le=1)
    baseline_pause_seconds: float = Field(0.2, ge=0, le=60)
    interruption_urgency_1_5: float = Field(2, ge=1, le=5)
    interruption_relevance_1_5: float = Field(3, ge=1, le=5)
    interruption_cognitive_demand_1_5: float = Field(3, ge=1, le=5)
    emotional_disturbance_1_5: float = Field(3, ge=1, le=5)
    task_type: str = "Programming"
    environment: str = "Multiple activities"
    deadline_pressure: str = "Low"
    interruption_type: str = "Phone call"
    interruption_expectedness: str = "Unexpected"
    condition: str = "Phone call"


class UserCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)


class BaselineCreate(BaseModel):
    wpm: float = Field(gt=0, le=300)
    error_rate: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    task_type: str = "Programming"


class EventCreate(Context):
    user_id: str
    source: str = "simulator"
    category: str = Field(min_length=1, max_length=80)


class OutcomeCreate(BaseModel):
    actual_duration_seconds: float = Field(gt=0, le=86400)
    returned_to_task: bool = True
    post_wpm: float | None = Field(default=None, gt=0, le=300)
    post_error_rate: float | None = Field(default=None, ge=0, le=1)


def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            created_at TEXT NOT NULL, consent_notification_access INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            created_at TEXT NOT NULL, wpm REAL NOT NULL, error_rate REAL NOT NULL,
            accuracy REAL NOT NULL, task_type TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL,
            source TEXT NOT NULL, category TEXT NOT NULL, context TEXT NOT NULL,
            predicted_duration REAL NOT NULL, prediction_kind TEXT NOT NULL,
            sample_count INTEGER NOT NULL, actual_duration REAL, returned_to_task INTEGER,
            post_wpm REAL, post_error_rate REAL
        );
        """
    )
    connection.commit()
    return connection


def user_or_404(connection: sqlite3.Connection, user_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return row


def latest_baseline(connection: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM baselines WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
    ).fetchone()


def model_duration(context: Context) -> tuple[float, float, str]:
    frame = pd.DataFrame([{feature: context.model_dump()[feature] for feature in FEATURES}])
    prediction = float(APP_BUNDLE["models"]["duration"].predict(frame)[0])
    return max(1.0, prediction), 0.0, "population_model"


def predict_event(event: EventCreate) -> dict[str, Any]:
    connection = db()
    user_or_404(connection, event.user_id)
    baseline = latest_baseline(connection, event.user_id)
    context = event.model_copy()
    if baseline:
        context.baseline_typing_wpm = baseline["wpm"]
        context.baseline_error_rate = baseline["error_rate"]
    base_duration, _, _ = model_duration(context)
    rows = connection.execute(
        "SELECT actual_duration FROM events WHERE user_id=? AND category=? "
        "AND actual_duration IS NOT NULL ORDER BY created_at DESC",
        (event.user_id, event.category),
    ).fetchall()
    user_values = [float(row["actual_duration"]) for row in rows]
    if len(user_values) >= 2:
        historical_mean = sum(user_values) / len(user_values)
        duration = 0.35 * base_duration + 0.65 * historical_mean
        kind = "personalized_category_history"
    else:
        duration = base_duration
        kind = "population_fallback" if not user_values else "provisional_personalized"
    duration = max(1.0, duration)
    wpm = float(APP_BUNDLE["models"]["typing_wpm"].predict(
        pd.DataFrame([{feature: context.model_dump()[feature] for feature in FEATURES}])
    )[0])
    error_rate = float(APP_BUNDLE["models"]["error_rate"].predict(
        pd.DataFrame([{feature: context.model_dump()[feature] for feature in FEATURES}])
    )[0])
    event_id = str(uuid.uuid4())
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)",
        (
            event_id, event.user_id, datetime.now(timezone.utc).isoformat(),
            event.source, event.category, json.dumps(event.model_dump()),
            duration, kind, len(user_values),
        ),
    )
    connection.commit()
    connection.close()
    return {
        "event_id": event_id, "duration_seconds": duration, "predicted_wpm": max(0, wpm),
        "predicted_error_rate": min(1, max(0, error_rate)),
        "prediction_kind": kind, "historical_sample_count": len(user_values),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/users")
def create_user(data: UserCreate) -> dict[str, str]:
    connection = db()
    user_id = str(uuid.uuid4())
    connection.execute(
        "INSERT INTO users(user_id,display_name,created_at) VALUES(?,?,?)",
        (user_id, data.display_name.strip(), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    connection.close()
    return {"user_id": user_id, "display_name": data.display_name.strip()}


@app.get("/api/users/{user_id}")
def get_user(user_id: str) -> dict[str, Any]:
    connection = db()
    row = user_or_404(connection, user_id)
    baseline = latest_baseline(connection, user_id)
    connection.close()
    return {"user_id": row["user_id"], "display_name": row["display_name"],
            "baseline": dict(baseline) if baseline else None}


@app.post("/api/users/{user_id}/baseline")
def save_baseline(user_id: str, data: BaselineCreate) -> dict[str, Any]:
    connection = db()
    user_or_404(connection, user_id)
    connection.execute(
        "INSERT INTO baselines(user_id,created_at,wpm,error_rate,accuracy,task_type) VALUES(?,?,?,?,?,?)",
        (user_id, datetime.now(timezone.utc).isoformat(), data.wpm, data.error_rate,
         data.accuracy, data.task_type),
    )
    connection.commit()
    connection.close()
    return {"status": "saved", "baseline": data.model_dump()}


@app.post("/api/events/predict")
def event_prediction(event: EventCreate) -> dict[str, Any]:
    return predict_event(event)


@app.post("/api/events/{event_id}/complete")
def complete_event(event_id: str, outcome: OutcomeCreate) -> dict[str, str]:
    connection = db()
    cursor = connection.execute(
        "UPDATE events SET actual_duration=?,returned_to_task=?,post_wpm=?,post_error_rate=? "
        "WHERE event_id=?",
        (outcome.actual_duration_seconds, int(outcome.returned_to_task),
         outcome.post_wpm, outcome.post_error_rate, event_id),
    )
    if cursor.rowcount == 0:
        connection.close()
        raise HTTPException(status_code=404, detail="Interruption event not found.")
    connection.commit()
    connection.close()
    return {"status": "recorded"}


@app.get("/api/users/{user_id}/history")
def user_history(user_id: str, limit: int = 30) -> list[dict[str, Any]]:
    connection = db()
    user_or_404(connection, user_id)
    rows = connection.execute(
        "SELECT event_id,created_at,source,category,predicted_duration,prediction_kind,"
        "sample_count,actual_duration,returned_to_task,post_wpm,post_error_rate "
        "FROM events WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, min(max(1, limit), 100)),
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


@app.delete("/api/users/{user_id}/data")
def delete_user_data(user_id: str) -> dict[str, str]:
    connection = db()
    user_or_404(connection, user_id)
    connection.execute("DELETE FROM events WHERE user_id=?", (user_id,))
    connection.execute("DELETE FROM baselines WHERE user_id=?", (user_id,))
    connection.execute("DELETE FROM users WHERE user_id=?", (user_id,))
    connection.commit()
    connection.close()
    return {"status": "deleted"}
