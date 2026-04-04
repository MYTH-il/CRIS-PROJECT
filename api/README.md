# CRIS API

Minimal FastAPI backend to decouple model logic from the Streamlit UI.

## Run
```powershell
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

## Endpoints
- `GET /health`
- `POST /token`
- `POST /users`
- `POST /classify`
- `POST /entities`
- `POST /similar`
- `POST /forecast/train`
- `POST /forecast`

## Database
Uses SQLite by default (`cris.db`). Override with `CRIS_DATABASE_URL`.

## Auth
Create a user, then request a token:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/users?username=admin&password=admin123&role=admin"
Invoke-RestMethod -Method Post "http://localhost:8000/token" -Body @{username="admin";password="admin123"} -ContentType "application/x-www-form-urlencoded"
```

Pass the returned bearer token in `Authorization: Bearer <token>`.

## Optional flags
Request models support `anonymize: true` to mask PII before processing.
