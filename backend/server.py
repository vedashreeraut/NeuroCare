import os
import sys
import shutil
import uuid
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlalchemy.orm import Session

# --------------------------------------------------
# PATHS AND PROJECT IMPORTS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import (
    VoiceModelRunner,
    AlcoholModelRunner,
    DEFAULT_SAMPLE_WAV,
)
from tremor_runner import TremorModelRunner

from backend.database import (
    init_db,
    get_db,
    User,
    AnalysisRecord,
    Notification,
)
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_doctor_user,
)
from backend.alert_engine import (
    evaluate_alert_rules,
    get_alert_settings,
)

# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

init_db()

# --------------------------------------------------
# PYDANTIC SCHEMAS
# Define these BEFORE registering API endpoints.
# --------------------------------------------------

class RegisterSchema(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "patient"


class LoginSchema(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Parkinson's Disease Multi-Modal Screening Dashboard API",
    description=(
        "Clinical decision-support and research demonstration API "
        "integrating voice, alcohol-demographic, and tremor models."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD ML MODELS
# --------------------------------------------------

print("Initializing backend ML model instances...")

voice_runner = VoiceModelRunner()
alcohol_runner = AlcoholModelRunner()
tremor_runner = TremorModelRunner()

print("All 3 ML Models loaded successfully in FastAPI Backend!")

# --------------------------------------------------
# UPLOAD DIRECTORY
# --------------------------------------------------

UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# AUDIO CONVERSION
# --------------------------------------------------


import os
import subprocess
from pathlib import Path

def convert_to_wav(file_path: str) -> str:
    """Convert uploaded audio to a valid mono, 16 kHz WAV file."""
    source = Path(file_path)
    output = source.with_name(source.stem + "_converted.wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(source),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0 or not output.exists():
            details = result.stderr[-1500:] if result.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"Audio conversion failed: {details}")

        return str(output)

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg is not installed. Install it using: brew install ffmpeg"
        )


# --------------------------------------------------
# DEMO USERS
# --------------------------------------------------

def seed_demo_users():
    db = next(get_db())

    try:
        doctor = (
            db.query(User)
            .filter(User.role == "doctor")
            .first()
        )

        if not doctor:
            db.add(
                User(
                    email="doctor@hospital.org",
                    password_hash=hash_password("DocPass123"),
                    full_name="Dr. Sarah Jenkins (Neurology)",
                    role="doctor",
                )
            )

        patient = (
            db.query(User)
            .filter(User.role == "patient")
            .first()
        )

        if not patient:
            db.add(
                User(
                    email="patient@example.com",
                    password_hash=hash_password("PatPass123"),
                    full_name="Arthur Pendelton",
                    role="patient",
                )
            )

        db.commit()

    except Exception as exc:
        db.rollback()
        print("Demo seed note:", exc)

    finally:
        db.close()


seed_demo_users()


# --------------------------------------------------
# AUTHENTICATION ENDPOINTS
# --------------------------------------------------

@app.post("/api/auth/register", response_model=TokenResponse)
def register(
    data: RegisterSchema,
    db: Session = Depends(get_db),
):
    existing = (
        db.query(User)
        .filter(User.email == str(data.email).lower())
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered.",
        )

    role = data.role.lower()

    if role not in ["doctor", "patient"]:
        role = "patient"

    user = User(
        email=str(data.email).lower(),
        password_hash=hash_password(data.password),
        full_name=data.full_name.strip(),
        role=role,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(
        data={"sub": str(user.id), "role": user.role}
    )

    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@app.post("/api/auth/login", response_model=TokenResponse)
def login(
    data: LoginSchema,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == data.email.lower())
        .first()
    )

    if not user or not verify_password(
        data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    token = create_access_token(
        data={"sub": str(user.id), "role": user.role}
    )

    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@app.get("/api/auth/me", response_model=UserResponse)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return UserResponse.model_validate(current_user)


@app.post("/api/auth/logout")
def logout():
    return {"message": "Successfully logged out."}


# --------------------------------------------------
# MAIN MULTI-MODAL ANALYSIS
# --------------------------------------------------

@app.post("/api/analysis/predict")
async def run_analysis(
    patient_name: str = Form(...),
    patient_age: float = Form(...),
    patient_gender: Optional[str] = Form("Unspecified"),
    avg_drinks: float = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    use_sample_audio: Optional[bool] = Form(False),
    tremor_segment: Optional[int] = Form(1),
    tremor_file: Optional[UploadFile] = File(None),
    use_sample_tremor: Optional[bool] = Form(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    print("=== Analysis Request ===")
    print(
        f"Patient: {patient_name}, Age: {patient_age}, "
        f"Drinks: {avg_drinks}"
    )
    print(
        f"Audio file: "
        f"{audio_file.filename if audio_file else 'None'}, "
        f"Use sample: {use_sample_audio}"
    )

    if not 18 <= patient_age <= 120:
        raise HTTPException(
            status_code=400,
            detail="Age must be between 18 and 120 years.",
        )

    if not 0 <= avg_drinks <= 25:
        raise HTTPException(
            status_code=400,
            detail="Alcohol drinks/day must be between 0 and 25.",
        )

    temp_audio_path = None
    temp_tremor_path = None
    saved_filename = "sample.wav"

    try:
        # 1. Voice model
        if audio_file and audio_file.filename:
            file_ext = Path(audio_file.filename).suffix.lower()

            allowed_extensions = {
                ".wav",
                ".mp3",
                ".flac",
                ".ogg",
                ".m4a",
                ".webm",
            }

            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsupported audio format. "
                        "Please upload WAV, MP3, FLAC, OGG, "
                        "M4A, or WebM audio."
                    ),
                )

            temp_filename = (
                f"rec_{uuid.uuid4().hex[:8]}{file_ext}"
            )
            original_path = UPLOADS_DIR / temp_filename

            with open(original_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            temp_audio_path = original_path
            saved_filename = (
                Path(audio_file.filename).stem + ".wav"
            )

            # Convert ALL uploads, including files incorrectly
            # labelled as WAV, into a valid WAV file.
            temp_audio_path = convert_to_wav(original_path)

        elif use_sample_audio or DEFAULT_SAMPLE_WAV.exists():
            temp_audio_path = DEFAULT_SAMPLE_WAV
            saved_filename = "recorded_sample_voice.wav"

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Audio recording or file upload is required."
                ),
            )

        voice_res = voice_runner.predict(temp_audio_path)

        # 2. Alcohol-demographic model
        alcohol_res = alcohol_runner.predict(
            patient_age,
            avg_drinks,
        )

        # 3. Tremor model
        if tremor_file and tremor_file.filename:
            tremor_ext = Path(
                tremor_file.filename
            ).suffix.lower()

            if tremor_ext != ".csv":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Uploaded tremor file must be a CSV "
                        "with sensor readings."
                    ),
                )

            temp_tremor_path = (
                UPLOADS_DIR
                / f"tremor_{uuid.uuid4().hex[:8]}.csv"
            )

            with open(temp_tremor_path, "wb") as buffer:
                shutil.copyfileobj(
                    tremor_file.file,
                    buffer,
                )

            tremor_res = tremor_runner.predict_csv(
                temp_tremor_path,
                source_name=tremor_file.filename,
            )

        elif (
            tremor_segment is not None
            and 1 <= tremor_segment <= tremor_runner.n_segments
        ):
            tremor_res = tremor_runner.predict_segment(
                int(tremor_segment)
            )

        else:
            tremor_res = tremor_runner.predict_segment(1)

        # 4. Save results in the database
        record = AnalysisRecord(
            user_id=current_user.id,
            patient_name=patient_name.strip(),
            patient_age=patient_age,
            patient_gender=patient_gender,
            avg_drinks=avg_drinks,
            voice_score=voice_res["combined_score"] * 100.0,
            voice_label=voice_res["predicted_label"],
            rf_score=voice_res["rf_score"] * 100.0,
            gb_score=voice_res["gb_score"] * 100.0,
            alcohol_prob=alcohol_res["probability_pct"],
            alcohol_risk=alcohol_res["risk_level"],
            tremor_score=tremor_res["combined_score"] * 100.0,
            tremor_label=tremor_res["predicted_label"],
            tremor_et_score=(
                tremor_res["extra_trees_score"] * 100.0
            ),
            tremor_rf_score=(
                tremor_res["random_forest_score"] * 100.0
            ),
            tremor_gb_score=(
                tremor_res["gradient_boosting_score"] * 100.0
            ),
            tremor_source=tremor_res.get(
                "source",
                "Benchmark Segment",
            ),
            alert_triggered=False,
            audio_filename=saved_filename,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        # 5. Evaluate alert rules
        alert_triggered = evaluate_alert_rules(
            db=db,
            patient_user=current_user,
            patient_name=patient_name.strip(),
            patient_age=patient_age,
            voice_confidence=record.voice_score,
            analysis_id=record.id,
        )

        if alert_triggered:
            record.alert_triggered = True
            db.commit()

        return {
            "analysis_id": record.id,
            "patient_name": record.patient_name,
            "patient_age": record.patient_age,
            "patient_gender": record.patient_gender,
            "avg_drinks": record.avg_drinks,
            "voice_analysis": {
                "prediction_label": voice_res["predicted_label"],
                "combined_score_pct": (
                    voice_res["combined_score"] * 100.0
                ),
                "rf_score_pct": voice_res["rf_score"] * 100.0,
                "gb_score_pct": voice_res["gb_score"] * 100.0,
                "predicted_class": voice_res["predicted_class"],
                "audio_source": saved_filename,
            },
            "alcohol_analysis": {
                "probability_pct": alcohol_res["probability_pct"],
                "risk_level": alcohol_res["risk_level"],
                "predicted_label": alcohol_res["predicted_label"],
            },
            "tremor_analysis": {
                "status": "Active / Verified",
                "available": True,
                "prediction_label": tremor_res["predicted_label"],
                "combined_score_pct": (
                    tremor_res["combined_score"] * 100.0
                ),
                "et_score_pct": (
                    tremor_res["extra_trees_score"] * 100.0
                ),
                "rf_score_pct": (
                    tremor_res["random_forest_score"] * 100.0
                ),
                "gb_score_pct": (
                    tremor_res["gradient_boosting_score"] * 100.0
                ),
                "predicted_class": tremor_res["predicted_class"],
                "windows_analysed": tremor_res["windows_analysed"],
                "source": tremor_res.get(
                    "source",
                    "Benchmark Segment",
                ),
                "rows": tremor_res.get("rows", "N/A"),
                "dataset_number": tremor_res.get("dataset_number"),
                "dataset_true_label": tremor_res.get(
                    "dataset_true_label"
                ),
                "match": tremor_res.get("match"),
            },
            "alert_triggered": alert_triggered,
            "timestamp": record.created_at.isoformat(),
        }

    except HTTPException:
        raise

    except Exception as exc:
        db.rollback()
        print(
            f"Analysis failed: {type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {type(exc).__name__}: {exc}"
            ),
        )

    finally:
        if (
            temp_audio_path
            and temp_audio_path != DEFAULT_SAMPLE_WAV
            and Path(temp_audio_path).exists()
        ):
            try:
                Path(temp_audio_path).unlink()
            except OSError:
                pass

        if temp_tremor_path and temp_tremor_path.exists():
            try:
                temp_tremor_path.unlink()
            except OSError:
                pass


# --------------------------------------------------
# STANDALONE TREMOR ANALYSIS
# --------------------------------------------------

@app.get("/api/tremor/catalog")
def get_tremor_catalog():
    return tremor_runner.get_segments_catalog()


@app.post("/api/tremor/predict")
async def run_standalone_tremor(
    tremor_segment: Optional[int] = Form(1),
    tremor_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
):
    temp_tremor_path = None

    try:
        if tremor_file and tremor_file.filename:
            tremor_ext = Path(
                tremor_file.filename
            ).suffix.lower()

            if tremor_ext != ".csv":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Uploaded tremor file must be in CSV format."
                    ),
                )

            temp_tremor_path = (
                UPLOADS_DIR
                / f"tremor_standalone_{uuid.uuid4().hex[:8]}.csv"
            )

            with open(temp_tremor_path, "wb") as buffer:
                shutil.copyfileobj(
                    tremor_file.file,
                    buffer,
                )

            return tremor_runner.predict_csv(
                temp_tremor_path,
                source_name=tremor_file.filename,
            )

        segment = tremor_segment or 1
        return tremor_runner.predict_segment(segment)

    finally:
        if temp_tremor_path and temp_tremor_path.exists():
            try:
                temp_tremor_path.unlink()
            except OSError:
                pass


# --------------------------------------------------
# ANALYSIS HISTORY
# --------------------------------------------------

@app.get("/api/analysis/history")
def get_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(AnalysisRecord)

    if current_user.role != "doctor":
        query = query.filter(
            AnalysisRecord.user_id == current_user.id
        )

    records = query.order_by(
        AnalysisRecord.created_at.desc()
    ).all()

    return [
        {
            "id": record.id,
            "user_id": record.user_id,
            "patient_name": record.patient_name,
            "patient_age": record.patient_age,
            "patient_gender": record.patient_gender,
            "avg_drinks": record.avg_drinks,
            "voice_score": record.voice_score,
            "voice_label": record.voice_label,
            "rf_score": record.rf_score,
            "gb_score": record.gb_score,
            "alcohol_prob": record.alcohol_prob,
            "alcohol_risk": record.alcohol_risk,
            "tremor_score": record.tremor_score,
            "tremor_label": record.tremor_label,
            "tremor_et_score": record.tremor_et_score,
            "tremor_rf_score": record.tremor_rf_score,
            "tremor_gb_score": record.tremor_gb_score,
            "tremor_source": record.tremor_source,
            "alert_triggered": record.alert_triggered,
            "audio_filename": record.audio_filename,
            "created_at": record.created_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        for record in records
    ]


@app.get("/api/analysis/history/{record_id}")
def get_analysis_detail(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.id == record_id)
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Analysis record not found.",
        )

    if (
        current_user.role != "doctor"
        and record.user_id != current_user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="Unauthorized access to this patient record.",
        )

    return {
        "id": record.id,
        "patient_name": record.patient_name,
        "patient_age": record.patient_age,
        "patient_gender": record.patient_gender,
        "avg_drinks": record.avg_drinks,
        "voice_score": record.voice_score,
        "voice_label": record.voice_label,
        "rf_score": record.rf_score,
        "gb_score": record.gb_score,
        "alcohol_prob": record.alcohol_prob,
        "alcohol_risk": record.alcohol_risk,
        "tremor_score": record.tremor_score,
        "tremor_label": record.tremor_label,
        "tremor_et_score": record.tremor_et_score,
        "tremor_rf_score": record.tremor_rf_score,
        "tremor_gb_score": record.tremor_gb_score,
        "tremor_source": record.tremor_source,
        "alert_triggered": record.alert_triggered,
        "audio_filename": record.audio_filename,
        "created_at": record.created_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


# --------------------------------------------------
# NOTIFICATIONS
# --------------------------------------------------

@app.get("/api/notifications")
def get_notifications(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db),
):
    notifications = (
        db.query(Notification)
        .order_by(Notification.created_at.desc())
        .all()
    )

    unread_count = (
        db.query(Notification)
        .filter(Notification.is_read == False)
        .count()
    )

    return {
        "unread_count": unread_count,
        "notifications": [
            {
                "id": notification.id,
                "patient_id": notification.patient_id,
                "patient_name": notification.patient_name,
                "patient_age": notification.patient_age,
                "voice_confidence": notification.voice_confidence,
                "analysis_id": notification.analysis_id,
                "is_read": notification.is_read,
                "created_at": notification.created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            for notification in notifications
        ],
    }


@app.patch("/api/notifications/{notif_id}/read")
def mark_notification_read(
    notif_id: int,
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db),
):
    notification = (
        db.query(Notification)
        .filter(Notification.id == notif_id)
        .first()
    )

    if not notification:
        raise HTTPException(
            status_code=404,
            detail="Notification not found.",
        )

    notification.is_read = True
    db.commit()

    return {
        "message": "Notification marked as read.",
        "id": notification.id,
    }


# --------------------------------------------------
# PATIENTS
# --------------------------------------------------

@app.get("/api/patients")
def get_patients(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db),
):
    patients = (
        db.query(User)
        .filter(User.role == "patient")
        .all()
    )

    return [
        {
            "id": patient.id,
            "full_name": patient.full_name,
            "email": patient.email,
            "total_analyses": (
                db.query(AnalysisRecord)
                .filter(
                    AnalysisRecord.user_id == patient.id
                )
                .count()
            ),
            "created_at": patient.created_at.strftime(
                "%Y-%m-%d"
            ),
        }
        for patient in patients
    ]


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

@app.get("/api/settings")
def get_settings():
    return get_alert_settings()


# --------------------------------------------------
# FRONTEND STATIC FILES
# --------------------------------------------------

FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )


@app.get("/")
def read_root():
    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))

    return {
        "message": "Parkinson's Dashboard API Server is Running!"
    }