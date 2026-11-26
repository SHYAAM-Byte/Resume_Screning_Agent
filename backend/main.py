
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import CounterSignal, DigitalSignal, Project, User, get_db, init_db
from processor import SignalProcessor
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set. Copy .env.example to .env and set it.")
JWT_ALGORITHM = "HS256"
TOKEN_HOURS = 8
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")
]
MAX_CAN_ID_LEN = 20  

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db() 
    yield

app = FastAPI(title="Vehicle Signal Data Processing", lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=CORS_ORIGINS,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://vehicle-signal-frontend-gdhmgvg5dtedgdfz.eastasia-01.azurewebsites.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(min_length=6, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt())
bearer = HTTPBearer(auto_error=False)


def _make_token(user: User) -> TokenOut:
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    token = jwt.encode({"sub": str(user.id), "exp": exp}, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenOut(access_token=token, username=user.username)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = db.get(User, int(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise unauthorized
    if user is None:
        raise unauthorized
    return user


@app.post("/api/auth/register", response_model=TokenOut, status_code=201)
def register(body: Credentials, db: Session = Depends(get_db)):
    password = body.password.encode()
    if len(password) > 72:  # bcrypt only uses the first 72 bytes
        raise HTTPException(400, "Password must be at most 72 bytes long")
    user = User(
        username=body.username.strip().lower(),  # case-insensitive usernames
        password_hash=bcrypt.hashpw(password, bcrypt.gensalt()).decode(),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Username already taken")
    return _make_token(user)


@app.post("/api/auth/login", response_model=TokenOut)
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username.strip().lower()))
    stored = user.password_hash.encode() if user else _DUMMY_HASH
    valid = bcrypt.checkpw(body.password.encode()[:72], stored)
    if not user or not valid:
        raise HTTPException(401, "Invalid username or password")
    return _make_token(user)


# ------------------------------------------------------------------ reports

def _build_report(db: Session, project: Project) -> dict:
    digital = db.scalars(
        select(DigitalSignal)
        .where(DigitalSignal.project_id == project.id)
        .order_by(DigitalSignal.can_id, DigitalSignal.byte_index, DigitalSignal.bit_index)
    ).all()
    counter = db.scalars(
        select(CounterSignal)
        .where(CounterSignal.project_id == project.id)
        .order_by(CounterSignal.can_id, CounterSignal.byte_index)
    ).all()
    return {
        "project": _project_dict(project),
        "digital_signals": [
            {
                "signal_type": "digital",
                "can_id": s.can_id,
                "byte": s.byte_index,
                "bit": s.bit_index,
                "transitions": s.transitions,
                "abs_time": s.abs_time,
                "values": s.value,
            }
            for s in digital
        ],
        "counter_signals": [
            {
                "signal_type": "counter",
                "can_id": s.can_id,
                "byte": s.byte_index,
                "min": s.min_value,
                "max": s.max_value,
                "step": s.step,
                "abs_time": s.abs_time,
                "values": s.value,
            }
            for s in counter
        ],
    }


def _project_dict(p: Project) -> dict:
    return {
        "id": p.id,
        "project_name": p.project_name,
        "vehicle_make": p.vehicle_make,
        "vehicle_model": p.vehicle_model,
        "vehicle_year": p.vehicle_year,
        "csv_filename": p.csv_filename,
        "row_count": p.row_count,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def _owned_project(db: Session, user: User, project_id: int) -> Project:
    project = db.scalar(
        select(Project).where(Project.id == project_id, Project.user_id == user.id)
    )
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@app.post("/api/projects/process")
def process_csv(
    file: UploadFile = File(...),
    project_name: str = Form(..., max_length=100),
    vehicle_make: str = Form(..., max_length=50),
    vehicle_model: str = Form(..., max_length=50),
    vehicle_year: int = Form(..., ge=1886, le=2100),
    digital: bool = Form(True),
    counter: bool = Form(True),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Process an uploaded CSV, store the project and detected signals, return the report."""
    project_name, vehicle_make, vehicle_model = (
        project_name.strip(),
        vehicle_make.strip(),
        vehicle_model.strip(),
    )
    if not (project_name and vehicle_make and vehicle_model):
        raise HTTPException(400, "Project name, make and model are required")
    if not (digital or counter):
        raise HTTPException(400, "Select at least one of Digital Signals / Counter Signals")
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")

    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"CSV is larger than {MAX_UPLOAD_MB} MB")

    try:
        result = SignalProcessor(file.file).process(digital=digital, counter=counter)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    signals = result["digital_signals"] + result["counter_signals"]
    if any(len(s["can_id"]) > MAX_CAN_ID_LEN for s in signals):
        raise HTTPException(400, f"CAN id longer than {MAX_CAN_ID_LEN} characters")

    project = Project(
        user_id=user.id,
        project_name=project_name,
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        vehicle_year=vehicle_year,
        csv_filename=filename[:255],
        row_count=result["row_count"],
    )
    db.add(project)
    db.flush()  # get project.id
    db.add_all(
        DigitalSignal(
            project_id=project.id,
            can_id=s["can_id"],
            byte_index=s["byte"],
            bit_index=s["bit"],
            transitions=s["transitions"],
            abs_time=s["abs_time"],
            value=s["values"],
        )
        for s in result["digital_signals"]
    )
    db.add_all(
        CounterSignal(
            project_id=project.id,
            can_id=s["can_id"],
            byte_index=s["byte"],
            min_value=s["min"],
            max_value=s["max"],
            step=s["step"],
            abs_time=s["abs_time"],
            value=s["values"],
        )
        for s in result["counter_signals"]
    )
    db.commit()

    report = _build_report(db, project)
    report["skipped_rows"] = result["skipped_rows"]
    return report


@app.get("/api/projects")
def list_projects(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """The user's previous projects (newest first) for View History."""
    projects = db.scalars(
        select(Project)
        .where(Project.user_id == user.id)
        .order_by(Project.created_at.desc(), Project.id.desc())
    ).all()
    ids = [p.id for p in projects]

    def counts(model) -> dict[int, int]:
        rows = db.execute(
            select(model.project_id, func.count())
            .where(model.project_id.in_(ids))
            .group_by(model.project_id)
        ).all()
        return {pid: n for pid, n in rows}

    n_digital, n_counter = counts(DigitalSignal), counts(CounterSignal)
    return [
        {
            **_project_dict(p),
            "digital_count": n_digital.get(p.id, 0),
            "counter_count": n_counter.get(p.id, 0),
        }
        for p in projects
    ]


@app.get("/api/projects/{project_id}")
def get_project(
    project_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Stored report for a previous project, without reprocessing the CSV."""
    return _build_report(db, _owned_project(db, user, project_id))
