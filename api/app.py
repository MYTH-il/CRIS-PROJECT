import os
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
from sqlalchemy.orm import Session

from src.classification import predict_crime_type, predict_zero_shot
from src.ner_extraction import extract_entities
from src.advanced_intel import AdvancedIntelligenceEngine
from src.predictive_mapping import PredictiveHotspotEngine
from src.semantic_search import SemanticSearchEngine
from src.pii_anonymizer import mask_pii
from .db import Base, engine, get_db
from .models import CaseRecord, User, RequestLog
from .auth import hash_password, verify_password, create_access_token
from jose import jwt


DATA_PATH = os.getenv("CRIS_DATA_PATH", os.path.join("data", "crime_reports_synthetic.csv"))

app = FastAPI(title="CRIS API", version="0.1.0")

Base.metadata.create_all(bind=engine)

_df_cache = None
_predictive_engine = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def _get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def _get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from jose import jwt
    from jose.exceptions import JWTError
    try:
        payload = jwt.decode(token, os.getenv("CRIS_SECRET_KEY", "change-me"), algorithms=["HS256"])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    user = _get_user(db, username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return user

def _load_df():
    global _df_cache
    if _df_cache is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        _df_cache = pd.read_csv(DATA_PATH)
    return _df_cache


class TextRequest(BaseModel):
    text: str
    model_type: Optional[str] = None
    use_zero_shot: bool = False
    anonymize: bool = False


class EntitiesRequest(BaseModel):
    text: str
    use_hf: bool = False
    anonymize: bool = False


class SimilarRequest(BaseModel):
    text: str
    method: str = "tfidf"
    top_n: int = 3
    anonymize: bool = False


class ForecastRequest(BaseModel):
    days: int = 1


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = _get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password.")
    access_token = create_access_token({"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users")
def create_user(username: str, password: str, role: str = "analyst", db: Session = Depends(get_db)):
    existing = _get_user(db, username)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists.")
    user = User(username=username, hashed_password=hash_password(password), role=role)
    db.add(user)
    db.commit()
    return {"status": "created", "username": username, "role": role}


@app.post("/classify")
def classify(req: TextRequest, db: Session = Depends(get_db), user: User = Depends(_get_current_user)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")
    text = mask_pii(req.text) if req.anonymize else req.text
    if req.use_zero_shot:
        df = _load_df()
        labels = list(df["crime_type"].dropna().unique()) if "crime_type" in df.columns else []
        label, score = predict_zero_shot(text, labels)
    else:
        label, score = predict_crime_type(text, model_type=req.model_type)
    if label is None:
        raise HTTPException(status_code=400, detail=score)
    # Optional: persist minimal record for audit trail
    record = CaseRecord(
        report_id=None,
        fir_number=None,
        report_date=None,
        crime_type=label,
        incident_description=text[:2000],
        incident_location=None,
    )
    db.add(record)
    db.commit()
    return {"label": label, "score": score}


@app.post("/entities")
def entities(req: EntitiesRequest, user: User = Depends(_get_current_user)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")
    text = mask_pii(req.text) if req.anonymize else req.text
    return {"entities": extract_entities(text, use_hf=req.use_hf)}


@app.post("/similar")
def similar(req: SimilarRequest, user: User = Depends(_get_current_user)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")
    text = mask_pii(req.text) if req.anonymize else req.text
    df = _load_df()
    if req.method == "semantic":
        engine = SemanticSearchEngine()
        ok, msg = engine.load_index()
        if ok:
            results, err = engine.query(text, top_n=req.top_n)
        else:
            results, err = engine.query_chroma(text, top_n=req.top_n)
        if err:
            raise HTTPException(status_code=400, detail=err)
        return {"results": results}

    intel = AdvancedIntelligenceEngine(DATA_PATH, os.path.join("models", "baseline_vectorizer.pkl"))
    results = intel.find_mo_similarity(text, top_n=req.top_n)
    return {"results": results or []}


@app.post("/forecast/train")
def train_forecast(user: User = Depends(_get_current_user)):
    global _predictive_engine
    df = _load_df()
    engine = PredictiveHotspotEngine()
    ok = engine.prepare_and_train(df)
    if not ok:
        raise HTTPException(status_code=400, detail="Missing required geospatial/temporal columns.")
    _predictive_engine = engine
    return {"status": "trained", "metrics": engine.metrics}


@app.post("/forecast")
def forecast(req: ForecastRequest, user: User = Depends(_get_current_user)):
    if req.days < 1 or req.days > 30:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 30.")
    if _predictive_engine is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /forecast/train first.")
    _, metrics, future_df = _predictive_engine.forecast_threat_map(target_days=req.days)
    daily = future_df.groupby("target_date")["predicted_count"].sum().reset_index()
    return {
        "metrics": metrics,
        "daily_forecast": daily.to_dict(orient="records"),
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.utcnow()
    if "predict-arrest" in request.url.path or "predict-offender" in request.url.path:
        return JSONResponse(status_code=403, content={"detail": "Arrest prediction is not permitted."})
    response = await call_next(request)
    duration = int((datetime.utcnow() - start).total_seconds() * 1000)
    try:
        db = next(get_db())
        user = "-"
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                payload = jwt.decode(auth.split(" ", 1)[1], os.getenv("CRIS_SECRET_KEY", "change-me"), algorithms=["HS256"])
                user = payload.get("sub", "authenticated")
            except Exception:
                user = "authenticated"
        db.add(RequestLog(
            path=request.url.path,
            method=request.method,
            user=user,
            status=response.status_code,
            duration_ms=duration,
        ))
        db.commit()
    except Exception:
        pass
    return response
