# Focusline: Personalized Cognitive Interruption System

This is a React + FastAPI research prototype for personalized interruption
prediction and recovery measurement using the HCSD dataset.

## Run locally

From `C:\PROJECTS\HCSD_Research`:

```powershell
& ".venv\Scripts\python.exe" "21_train_complete_model.py"
& ".venv\Scripts\python.exe" "22_train_application_models.py"
& ".venv\Scripts\python.exe" -m uvicorn app.backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
Set-Location app\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The API stores profiles, baselines, simulated interruption events, predictions,
and measured outcomes in `app/prediction_history.sqlite3`. The timer and browser
notification run in the browser.

## Product workflow

1. Create a local user profile.
2. Save a typing baseline from a short, consent-based assessment.
3. Use **Test event** to simulate an incoming application notification.
4. Start the countdown using the model's predicted duration.
5. Record the actual time away and optional post-event typing assessment.
6. Repeat. After two measured outcomes for a category, that user's category
   history is blended with the population model.

The simulator is intentionally labelled. A normal React website cannot silently
read Android, Windows, WhatsApp, Instagram, or other application notifications.
Real integrations require a separately permissioned browser extension or native
companion and are not included in this prototype.

## Model boundaries and targets

The application predicts observed dataset targets:

- interruption duration,
- first recovery-window typing speed,
- first recovery-window error rate,
- the existing synthetic recovery-time objective.

The application-facing duration model uses `interruption_duration_seconds` as
the observed synthetic duration target, but this is not proof of real-world
return-to-task behavior. User-specific predictions are only labelled
`personalized_category_history` after at least two measured outcomes for that
user and category. Otherwise the API returns `population_fallback` or
`provisional_personalized`.

It does not claim to predict a verified remaining recovery time because the
dataset has no independent interruption-end and task-resumption timestamps.
The dashboard labels values as estimates and includes a synthetic-data
disclaimer. A future evaluation of fixed versus personalized reminders must
collect measured resumption outcomes; prediction history alone is not evidence
of effectiveness.

## Stored data and privacy

The SQLite prototype stores a generated user ID, display name, baseline values,
event category/source, model prediction, and optional measured outcomes. It
does not store notification message contents. The History page provides a
delete-my-data action. Notification permission is requested only when the user
triggers the simulator and can be disabled in the dashboard.
