from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal

import jwt
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from database import (
    ChatMessageModel,
    ChatSessionModel,
    RagUserModel,
    AuthSessionLocal,
    SessionLocal,
    ensure_chat_schema,
    session_for_prisma_reads,
)
from scripts.chat import HISTORY_MESSAGES, RAGAssistant

logger = logging.getLogger("rag")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


JWT_SECRET = os.environ.get("JWT_SECRET", "").strip()
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "").strip() or None
JWT_ISSUER = os.environ.get("JWT_ISSUER", "").strip() or None
# Used only when JWT_SECRET is unset so local register/login can still issue tokens
# that /sessions will accept. Set JWT_SECRET in any exposed environment.
_DEV_JWT_SECRET = "dev-jwt-secret-do-not-use-in-production"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))
AUTH_COOKIE_NAME = "access_token"
ASSISTANT_UNAVAILABLE_DETAIL = (
    "Assistant unavailable (model/embeddings failed to load). "
    "On low-memory hosts set LLM_PROVIDER=hf_space or RAG_REMOTE_LLM_URL."
)


def _assistant_error_detail(exc: BaseException) -> str:
    message = str(exc).strip()
    return message[:800] if message else ASSISTANT_UNAVAILABLE_DETAIL
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PBKDF2_ITERATIONS = 210_000

bearer_scheme = HTTPBearer(auto_error=False)


def _jwt_signing_secret() -> str:
    return JWT_SECRET or _DEV_JWT_SECRET


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations_s, salt, digest_hex = stored.split("$", 3)
        iterations = int(iterations_s)
    except (ValueError, TypeError):
        return False
    if algo != "pbkdf2_sha256" or iterations < 1:
        return False
    expected = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
    )
    return hmac.compare_digest(expected.hex(), digest_hex)


def create_access_token(*, user_id: str, email: str, name: str) -> str:
    now = utcnow()
    payload: Dict[str, Any] = {
        "sub": user_id,
        "id": user_id,
        "email": email,
        "name": name,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    if JWT_AUDIENCE:
        payload["aud"] = JWT_AUDIENCE
    if JWT_ISSUER:
        payload["iss"] = JWT_ISSUER
    return jwt.encode(payload, _jwt_signing_secret(), algorithm=JWT_ALGORITHM)


def _decode_access_token(token: str) -> str:
    decode_kwargs: Dict[str, Any] = {"algorithms": [JWT_ALGORITHM]}
    if JWT_AUDIENCE:
        decode_kwargs["audience"] = JWT_AUDIENCE
    if JWT_ISSUER:
        decode_kwargs["issuer"] = JWT_ISSUER
    try:
        payload = jwt.decode(token, _jwt_signing_secret(), **decode_kwargs)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e
    user_id = payload.get("sub") or payload.get("id")
    if not user_id or not str(user_id).strip():
        raise HTTPException(status_code=401, detail="Token missing 'sub' (or 'id') claim.")
    return str(user_id).strip()


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    access_token_cookie: Annotated[str | None, Cookie(alias=AUTH_COOKIE_NAME)] = None,
) -> str:
    token: str | None = None
    if credentials is not None and credentials.scheme.lower() == "bearer" and credentials.credentials:
        token = credentials.credentials
    elif access_token_cookie and access_token_cookie.strip():
        token = access_token_cookie.strip()

    if token:
        return _decode_access_token(token)

    if JWT_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header (Bearer token required).",
        )

    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    raise HTTPException(
        status_code=401,
        detail="Set JWT_SECRET and send Authorization: Bearer <jwt>, or for local dev send X-User-Id.",
    )


def _cookie_secure(request: Request) -> bool:
    raw = os.environ.get("COOKIE_SECURE", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return request.url.scheme == "https"


def _set_auth_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        max_age=JWT_EXPIRE_MINUTES * 60,
        path="/",
    )


def _clear_auth_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
    )


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    name: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        email = value.strip().lower()
        if not _EMAIL_RE.match(email):
            raise ValueError("Enter a valid email address.")
        return email

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        name = " ".join(value.split())
        if len(name) < 2:
            raise ValueError("Name must be at least 2 characters.")
        if len(name) > 255:
            raise ValueError("Name is too long.")
        return name


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(BaseModel):
    id: str
    email: str
    name: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def _user_out(user: RagUserModel) -> UserOut:
    return UserOut(id=user.id, email=user.email, name=user.name or None)


def _auth_response(user: RagUserModel) -> AuthResponse:
    token = create_access_token(user_id=user.id, email=user.email, name=user.name or "")
    return AuthResponse(access_token=token, token_type="bearer", user=_user_out(user))


def _get_user_by_id(user_id: str) -> RagUserModel | None:
    with AuthSessionLocal() as db:
        return db.get(RagUserModel, user_id)


class VehicleHealthComponent(BaseModel):
    label: str
    percent: int = Field(ge=0, le=100)


class VehicleHealthPayload(BaseModel):
    overallPercent: int = Field(ge=0, le=100)
    summary: str | None = None
    components: List[VehicleHealthComponent] = Field(default_factory=list)


class VehiclePayload(BaseModel):
    id: str
    driverId: str | None = None
    plateNumber: str | None = None
    make: str | None = None
    model: str | None = None
    year: int | None = None
    displayName: str | None = None
    type: str | None = None
    color: str | None = None
    vin: str | None = None
    mileage: int | None = None
    fuelType: str | None = None
    status: str | None = None
    imageUrl: str | None = None
    insuranceDocumentUrl: str | None = None
    insuranceExpiresAt: datetime | None = None
    registrationDocumentUrl: str | None = None
    registrationExpiresAt: datetime | None = None
    createdAt: datetime | None = None
    updatedAt: datetime | None = None


class VehicleContextPayload(BaseModel):
    vehicle: VehiclePayload | None = None
    vehicle_health: VehicleHealthPayload | None = None
    snapshot_at: datetime | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default="New chat", max_length=255)
    vehicle_id: str | None = None
    car_context: str = Field(default="", max_length=255)
    vehicle_context: VehicleContextPayload | None = None
    vehicle: VehiclePayload | None = None
    vehicle_health: VehicleHealthPayload | None = None


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)
    car_context: str = Field(default="", max_length=255)
    vehicle_id: str | None = None
    title: str | None = None
    vehicle_context: VehicleContextPayload | None = None
    vehicle: VehiclePayload | None = None
    vehicle_health: VehicleHealthPayload | None = None
    use_user_manual: bool = True


class ChatSessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    car_context: str
    vehicle_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime


_assistant_cache: Dict[str, RAGAssistant] = {}


def build_assistant_key(user_id: str, _car_context: str, use_user_manual: bool) -> str:
    # Vehicle text is passed into generate_answer on each request. Keying the
    # cache on it built a new assistant (and reloaded the LLM) whenever it changed.
    return f"{user_id}::{use_user_manual}"


def get_assistant(user_id: str, car_context: str, use_user_manual: bool) -> RAGAssistant:
    key = build_assistant_key(user_id, car_context, use_user_manual)
    cached = _assistant_cache.get(key)
    if cached is not None:
        return cached

    assistant = RAGAssistant(
        user_id=user_id,
        car_context=car_context,
        use_user_manual=use_user_manual,
    )
    _assistant_cache[key] = assistant
    return assistant


def to_session_response(session: ChatSessionModel) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        car_context=session.car_context,
        vehicle_id=getattr(session, "vehicle_id", None),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def to_message_response(message: ChatMessageModel) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _shorten_context(text: str, max_len: int = 255) -> str:
    text = " ".join(text.split()).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def normalize_vehicle_context(payload: Any) -> tuple[str, str]:
    """Return (short_context_for_session, detailed_priority_context_for_retrieval)."""
    vehicle_context = getattr(payload, "vehicle_context", None)
    vehicle = (vehicle_context.vehicle if vehicle_context else None) or getattr(payload, "vehicle", None)
    health = (vehicle_context.vehicle_health if vehicle_context else None) or getattr(
        payload, "vehicle_health", None
    )
    manual_context = (getattr(payload, "car_context", "") or "").strip()

    parts: List[str] = []
    if vehicle is not None:
        label = (
            vehicle.displayName
            or " ".join(
                p for p in [str(vehicle.year) if vehicle.year else "", vehicle.make or "", vehicle.model or ""]
                if p
            ).strip()
        )
        if label:
            parts.append(f"Vehicle: {label}")
        if vehicle.plateNumber:
            parts.append(f"Plate: {vehicle.plateNumber}")
        if vehicle.fuelType:
            parts.append(f"Fuel: {vehicle.fuelType}")
        if vehicle.mileage is not None:
            parts.append(f"Mileage: {vehicle.mileage} km")

    priority_lines: List[str] = []
    if health is not None:
        priority_lines.append(f"Overall maintenance health: {health.overallPercent}%")
        if health.summary:
            priority_lines.append(f"Summary: {health.summary.strip()}")
        for c in sorted(health.components, key=lambda x: (x.label or "").lower()):
            priority_lines.append(f"{c.label}: {c.percent}%")
        concerning = [c for c in health.components if c.percent <= 60]
        if concerning:
            issues = ", ".join(f"{c.label} at {c.percent}%" for c in concerning)
            priority_lines.append(f"Needs attention: {issues}")

    if manual_context:
        parts.append(f"Driver context: {manual_context}")
        priority_lines.append(f"Driver context: {manual_context}")

    short_context = _shorten_context(" | ".join(parts) if parts else manual_context)
    detailed_context = "\n".join(priority_lines + parts).strip()
    return short_context, detailed_context


def _extract_vehicle_id(payload: Any) -> str:
    if getattr(payload, "vehicle_id", None):
        return str(payload.vehicle_id).strip()
    vehicle_context = getattr(payload, "vehicle_context", None)
    if vehicle_context and vehicle_context.vehicle and vehicle_context.vehicle.id:
        return str(vehicle_context.vehicle.id).strip()
    if getattr(payload, "vehicle", None) and payload.vehicle.id:
        return str(payload.vehicle.id).strip()
    return ""


def _health_from_db(raw_health: Any) -> VehicleHealthPayload | None:
    if not isinstance(raw_health, dict):
        return None

    numeric_components: List[VehicleHealthComponent] = []
    for key, value in raw_health.items():
        if key == "custom":
            continue
        if isinstance(value, (int, float)):
            pct = max(0, min(100, int(value)))
            numeric_components.append(
                VehicleHealthComponent(label=str(key).replace("_", " ").title(), percent=pct)
            )

    overall = int(round(sum(c.percent for c in numeric_components) / len(numeric_components))) if numeric_components else 0
    summary = f"{len([c for c in numeric_components if c.percent <= 60])} component(s) need attention."
    return VehicleHealthPayload(
        overallPercent=overall,
        summary=summary,
        components=numeric_components,
    )


def fetch_vehicle_context_from_db(vehicle_id: str, user_id: str) -> tuple[VehiclePayload | None, VehicleHealthPayload | None]:
    if not vehicle_id:
        return None, None

    vehicle_sql_owner = """
    SELECT
      id, "driverId", "plateNumber", make, model, year, type, color, vin, mileage,
      fuel_type, image_url, insurance_document_url, insurance_expires_at,
      registration_document_url, registration_expires_at, "createdAt", "updatedAt"
    FROM "Vehicle"
    WHERE id = :vehicle_id
      AND "driverId" = :user_id
    LIMIT 1
    """
    vehicle_sql_by_id = """
    SELECT
      id, "driverId", "plateNumber", make, model, year, type, color, vin, mileage,
      fuel_type, image_url, insurance_document_url, insurance_expires_at,
      registration_document_url, registration_expires_at, "createdAt", "updatedAt"
    FROM "Vehicle"
    WHERE id = :vehicle_id
    LIMIT 1
    """
    health_sql = """
    SELECT health
    FROM "VehicleMaintenanceHealth"
    WHERE "vehicleId" = :vehicle_id
    ORDER BY "updatedAt" DESC
    LIMIT 1
    """

    try:
        with SessionLocal() as db:
            row = db.execute(
                text(vehicle_sql_owner), {"vehicle_id": vehicle_id, "user_id": user_id}
            ).mappings().first()
            if row is None:
                # Dev fallback: allow vehicle lookup by id when X-User-Id does not match Driver.id yet.
                row = db.execute(text(vehicle_sql_by_id), {"vehicle_id": vehicle_id}).mappings().first()
            if row is None:
                return None, None

            rd = _row_as_dict(row)
            vehicle = VehiclePayload(
                id=str(rd["id"]),
                driverId=str(rd["driverId"]) if rd.get("driverId") is not None else None,
                plateNumber=rd.get("plateNumber"),
                make=rd.get("make"),
                model=rd.get("model"),
                year=rd.get("year"),
                displayName=" ".join(
                    p
                    for p in [
                        str(rd.get("year")) if rd.get("year") else "",
                        rd.get("make") or "",
                        rd.get("model") or "",
                    ]
                    if p
                ).strip()
                or None,
                type=rd.get("type"),
                color=rd.get("color"),
                vin=rd.get("vin"),
                mileage=rd.get("mileage"),
                fuelType=rd.get("fuel_type"),
                imageUrl=rd.get("image_url"),
                insuranceDocumentUrl=rd.get("insurance_document_url"),
                insuranceExpiresAt=rd.get("insurance_expires_at"),
                registrationDocumentUrl=rd.get("registration_document_url"),
                registrationExpiresAt=rd.get("registration_expires_at"),
                createdAt=rd.get("createdAt"),
                updatedAt=rd.get("updatedAt"),
            )

            health_row = db.execute(text(health_sql), {"vehicle_id": vehicle_id}).mappings().first()
            hd = _row_as_dict(health_row) if health_row is not None else {}
            health = _health_from_db(hd["health"]) if hd.get("health") is not None else None
            return vehicle, health
    except SQLAlchemyError:
        # Chat DB may be SQLite or missing Prisma tables (Vehicle / VehicleMaintenanceHealth).
        return None, None


def resolve_context_hybrid(
    payload: Any, user_id: str, *, fallback_vehicle_id: str | None = None
) -> tuple[str, str]:
    vehicle_id = (_extract_vehicle_id(payload) or "").strip() or (fallback_vehicle_id or "").strip()
    vehicle_db, health_db = fetch_vehicle_context_from_db(vehicle_id, user_id)

    vehicle_context = getattr(payload, "vehicle_context", None)
    vehicle_flutter = (vehicle_context.vehicle if vehicle_context else None) or getattr(payload, "vehicle", None)
    health_flutter = (vehicle_context.vehicle_health if vehicle_context else None) or getattr(
        payload, "vehicle_health", None
    )

    # Hybrid: DB is canonical when available; otherwise use Flutter payload.
    vehicle = vehicle_db or vehicle_flutter
    health = health_db or health_flutter
    manual_context = (getattr(payload, "car_context", "") or "").strip()

    shadow = type("ContextShadow", (), {})()
    shadow.vehicle_context = type("VC", (), {"vehicle": vehicle, "vehicle_health": health})()
    shadow.vehicle = vehicle
    shadow.vehicle_health = health
    shadow.car_context = manual_context
    return normalize_vehicle_context(shadow)


def _norm_token(value: str | None) -> str:
    if not value:
        return ""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in str(value).lower()).split())


def _row_as_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def resolve_candidate_manual_ids(
    *,
    make: str | None,
    model: str | None,
    year: int | None,
    car_context: str,
    limit: int = 4,
) -> List[str]:
    """
    Rank likely manual ids from EducationContent using vehicle identity.
    Strategy: exact make+model+year, then make+model, then make/context fallback.
    """
    mk = _norm_token(make)
    mdl = _norm_token(model)
    yr = str(year).strip() if year else ""
    ctx = _norm_token(car_context)
    if not any([mk, mdl, yr, ctx]):
        return []

    # Query by title first because uploaded manuals are titled with make/model/year.
    sql = """
    SELECT id, title
    FROM "EducationContent"
    WHERE pdf_url IS NOT NULL
      AND category::text = 'MANUALS'
      AND (
        (:mk = '' OR lower(title) LIKE '%' || :mk || '%')
        OR (:mdl = '' OR lower(title) LIKE '%' || :mdl || '%')
        OR (:yr = '' OR lower(title) LIKE '%' || :yr || '%')
        OR (:ctx = '' OR lower(title) LIKE '%' || :ctx || '%')
      )
    """
    try:
        with session_for_prisma_reads() as db:
            rows = db.execute(text(sql), {"mk": mk, "mdl": mdl, "yr": yr, "ctx": ctx}).mappings().all()
    except SQLAlchemyError:
        # Wrong dialect (e.g. SQLite), missing EducationContent, or column mismatch.
        return []

    def score(title: str) -> tuple[int, int]:
        t = _norm_token(title)
        exact = int(bool(mk and mk in t)) + int(bool(mdl and mdl in t)) + int(bool(yr and yr in t))
        broad = int(bool(ctx and ctx in t))
        return exact, broad

    ranked = sorted(
        rows,
        key=lambda r: score(str(_row_as_dict(r).get("title") or "")),
        reverse=True,
    )
    out: List[str] = []
    for r in ranked:
        rid = str(_row_as_dict(r).get("id") or "").strip()
        if rid and rid not in out:
            out.append(rid)
        if len(out) >= limit:
            break
    return out


def get_owned_session_or_404(session_id: str, user_id: str) -> ChatSessionModel:
    with SessionLocal() as db:
        session = db.get(ChatSessionModel, session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        return session


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    ensure_chat_schema()
    yield


app = FastAPI(title="RAG Chat History API", version="1.0.0", lifespan=_lifespan)


@app.middleware("http")
async def unhandled_exception_logger(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        dbg = os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "yes")
        payload: Dict[str, Any] = {"detail": "Internal server error"}
        if dbg:
            payload["detail"] = str(exc)
            payload["type"] = type(exc).__name__
            payload["trace"] = traceback.format_exc()[-8000:]
        return JSONResponse(status_code=500, content=payload)


# Registered after the error middleware so it stays outermost. Otherwise a 500
# leaves the browser with no Access-Control-Allow-Origin header.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "RAG Chat History API", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest, request: Request, response: Response) -> AuthResponse:
    user = RagUserModel(
        id=str(uuid.uuid4()),
        email=payload.email,
        password_hash=hash_password(payload.password),
        name=payload.name,
        created_at=utcnow(),
    )
    try:
        with AuthSessionLocal() as db:
            db.add(user)
            db.commit()
            db.refresh(user)
            stored = RagUserModel(
                id=user.id,
                email=user.email,
                password_hash=user.password_hash,
                name=user.name,
            )
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Email already registered") from None

    body = _auth_response(stored)
    _set_auth_cookie(response, request, body.access_token)
    return body


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> AuthResponse:
    with AuthSessionLocal() as db:
        user = db.scalars(select(RagUserModel).where(RagUserModel.email == payload.email)).first()
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        stored = RagUserModel(
            id=user.id,
            email=user.email,
            password_hash=user.password_hash,
            name=user.name,
        )

    body = _auth_response(stored)
    _set_auth_cookie(response, request, body.access_token)
    return body


@app.get("/auth/me", response_model=UserOut)
def auth_me(user_id: Annotated[str, Depends(get_current_user_id)]) -> UserOut:
    user = _get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return _user_out(user)


@app.post("/auth/logout")
def logout(request: Request, response: Response) -> Dict[str, bool]:
    _clear_auth_cookie(response, request)
    return {"ok": True}


@app.post("/sessions", response_model=ChatSessionResponse)
def create_session(
    payload: CreateSessionRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> ChatSessionResponse:
    vid = (_extract_vehicle_id(payload) or "").strip() or None
    short_ctx, _ = resolve_context_hybrid(payload, user_id, fallback_vehicle_id=vid)
    now = utcnow()
    session = ChatSessionModel(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=((payload.title or "").strip() or "New chat"),
        car_context=short_ctx,
        vehicle_id=vid,
        created_at=now,
        updated_at=now,
    )
    with SessionLocal() as db:
        db.add(session)
        db.commit()
        db.refresh(session)
    return to_session_response(session)


@app.get("/sessions", response_model=List[ChatSessionResponse])
def list_sessions(user_id: Annotated[str, Depends(get_current_user_id)]) -> List[ChatSessionResponse]:
    with SessionLocal() as db:
        rows = db.scalars(
            select(ChatSessionModel)
            .where(ChatSessionModel.user_id == user_id)
            .order_by(desc(ChatSessionModel.updated_at))
        ).all()
    return [to_session_response(s) for s in rows]


@app.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> ChatSessionResponse:
    session = get_owned_session_or_404(session_id, user_id)
    return to_session_response(session)


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> Response:
    with SessionLocal() as db:
        session = db.get(ChatSessionModel, session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        db.delete(session)
        db.commit()
    return Response(status_code=204)


@app.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
def list_messages(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    limit: int = Query(default=50, ge=1, le=200),
) -> List[ChatMessageResponse]:
    with SessionLocal() as db:
        owns = db.scalar(
            select(ChatSessionModel.id)
            .where(ChatSessionModel.id == session_id, ChatSessionModel.user_id == user_id)
            .limit(1)
        )
        if owns is None:
            raise HTTPException(status_code=404, detail="Session not found")
        q = (
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(desc(ChatMessageModel.created_at), desc(ChatMessageModel.id))
            .limit(limit)
        )
        rows = list(db.scalars(q).all())

    items_chrono = list(reversed(rows))
    return [to_message_response(m) for m in items_chrono]


@dataclass
class _TurnContext:
    """State shared by the JSON and streaming chat endpoints, built before generation starts."""

    session: ChatSessionModel
    assistant: RAGAssistant
    user_message: ChatMessageModel
    user_text: str
    prev_summary: str
    recent_messages: List[Dict[str, str]]
    active_context: str
    detailed_ctx: str
    manual_ids: List[str]
    merged_vid: str
    payload_vehicle_id: str
    session_title: str | None


def _prepare_turn(
    session_id: str,
    payload: ChatMessageRequest,
    user_id: str,
) -> _TurnContext:
    """Persist the user message and resolve everything the assistant needs to answer."""
    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    payload_vehicle_id = (_extract_vehicle_id(payload) or "").strip()

    with SessionLocal() as db:
        session = db.get(ChatSessionModel, session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        prev_summary = (getattr(session, "chat_summary", "") or "").strip()

        user_message = ChatMessageModel(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=text,
            created_at=utcnow(),
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)

        # Last few messages for coherence (chronological).
        recent_rows = list(
            db.scalars(
                select(ChatMessageModel)
                .where(ChatMessageModel.session_id == session_id)
                .order_by(desc(ChatMessageModel.created_at), desc(ChatMessageModel.id))
                .limit(HISTORY_MESSAGES)
            ).all()
        )
        recent_rows = list(reversed(recent_rows))
        recent_messages = [{"role": m.role, "content": m.content} for m in recent_rows]

    merged_vid = payload_vehicle_id or (session.vehicle_id or "").strip()
    short_ctx, detailed_ctx = resolve_context_hybrid(
        payload, user_id, fallback_vehicle_id=merged_vid or None
    )
    active_context = short_ctx or session.car_context
    vehicle_db, _ = fetch_vehicle_context_from_db(merged_vid, user_id)
    vehicle_context = getattr(payload, "vehicle_context", None)
    vehicle_flutter = (vehicle_context.vehicle if vehicle_context else None) or getattr(payload, "vehicle", None)
    vehicle_for_manual = vehicle_db or vehicle_flutter
    manual_ids = resolve_candidate_manual_ids(
        make=(vehicle_for_manual.make if vehicle_for_manual else None),
        model=(vehicle_for_manual.model if vehicle_for_manual else None),
        year=(vehicle_for_manual.year if vehicle_for_manual else None),
        car_context=active_context,
    )
    if payload.title and payload.title.strip():
        session_title = payload.title.strip()
    else:
        session_title = None

    try:
        assistant = get_assistant(
            user_id=session.user_id,
            car_context=active_context,
            use_user_manual=payload.use_user_manual,
        )
    except RuntimeError as exc:
        # Typical on Render: local LLM weights failed to load (set LLM_PROVIDER=hf_space or RAG_REMOTE_LLM_URL).
        logger.exception("RAG assistant failed: %s", exc)
        raise HTTPException(status_code=503, detail=_assistant_error_detail(exc)) from exc

    return _TurnContext(
        session=session,
        assistant=assistant,
        user_message=user_message,
        user_text=text,
        prev_summary=prev_summary,
        recent_messages=recent_messages,
        active_context=active_context,
        detailed_ctx=detailed_ctx,
        manual_ids=manual_ids,
        merged_vid=merged_vid,
        payload_vehicle_id=payload_vehicle_id,
        session_title=session_title,
    )


def _persist_turn(ctx: _TurnContext, session_id: str, answer: str) -> Dict[str, Any]:
    """Store the assistant reply and roll the session forward. Mirrors the pre-streaming behaviour."""
    with SessionLocal() as db:
        assistant_message = ChatMessageModel(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=answer,
            created_at=utcnow(),
        )
        db.add(assistant_message)

        session_to_update = db.get(ChatSessionModel, session_id)
        if session_to_update is not None:
            session_to_update.updated_at = utcnow()
            if ctx.active_context:
                session_to_update.car_context = _shorten_context(ctx.active_context)
            # Rolling summary: update once per assistant reply.
            try:
                session_to_update.chat_summary = ctx.assistant.update_chat_summary(
                    ctx.prev_summary, ctx.user_text, answer
                )
            except Exception:
                pass
            if ctx.payload_vehicle_id:
                session_to_update.vehicle_id = ctx.payload_vehicle_id
            elif ctx.merged_vid and not (session_to_update.vehicle_id or "").strip():
                session_to_update.vehicle_id = ctx.merged_vid
            if ctx.session_title:
                session_to_update.title = ctx.session_title

        db.commit()
        db.refresh(assistant_message)
        refreshed_session = db.get(ChatSessionModel, session_id)

    return {
        "session": to_session_response(
            refreshed_session if refreshed_session is not None else ctx.session
        ),
        "user_message": to_message_response(ctx.user_message),
        "assistant_message": to_message_response(assistant_message),
    }


@app.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    payload: ChatMessageRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> Dict[str, Any]:
    ctx = _prepare_turn(session_id, payload, user_id)
    try:
        answer = ctx.assistant.generate_answer(
            ctx.user_text,
            car_context=ctx.active_context,
            priority_context=ctx.detailed_ctx,
            manual_ids=ctx.manual_ids,
            chat_summary=ctx.prev_summary,
            recent_messages=ctx.recent_messages,
        )
    except RuntimeError as exc:
        logger.exception("RAG assistant failed: %s", exc)
        raise HTTPException(status_code=503, detail=_assistant_error_detail(exc)) from exc

    return _persist_turn(ctx, session_id, answer)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/sessions/{session_id}/messages/stream")
def stream_message(
    session_id: str,
    payload: ChatMessageRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> StreamingResponse:
    """Server-sent events variant of `send_message`.

    Emits `delta` events as the model writes, then a single `done` event carrying the same body
    the JSON endpoint returns. Deltas are provisional — answer post-processing can rewrite or
    replace the reply — so clients should render the `done` text in place of accumulated deltas.
    """
    ctx = _prepare_turn(session_id, payload, user_id)

    def event_stream():
        try:
            final_answer = ""
            for kind, chunk in ctx.assistant.generate_answer_stream(
                ctx.user_text,
                car_context=ctx.active_context,
                priority_context=ctx.detailed_ctx,
                manual_ids=ctx.manual_ids,
                chat_summary=ctx.prev_summary,
                recent_messages=ctx.recent_messages,
            ):
                if kind == "delta":
                    yield _sse("delta", {"text": chunk})
                else:
                    final_answer = chunk
            body = _persist_turn(ctx, session_id, final_answer)
            yield _sse("done", {k: v.model_dump(mode="json") for k, v in body.items()})
        except Exception as exc:
            # The response has already started, so the status code cannot change; report in-band.
            logger.exception("Streaming chat failed: %s", exc)
            dbg = os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "yes")
            yield _sse("error", {"detail": str(exc) if dbg else _assistant_error_detail(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx and similar proxies from buffering the stream into one response.
            "X-Accel-Buffering": "no",
        },
    )
