"""
Pairwise Evaluation Web App
===========================
FastAPI backend for an interactive multi-turn pairwise comparison study.
Each participant chats with two assistant models side-by-side on a writing
task, picks the preferred reply at every turn (with a short rationale), and
both models continue from the chosen reply. Exactly two models are compared
per conversation (the two configured by the admin).

Start with:
  uvicorn app:app --host 127.0.0.1 --port 7860 --proxy-headers --forwarded-allow-ips='*'

See README.md for the full deployment guide.
"""
from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response

from generation_utils import generate_pair, get_client
from writing_task_bank import (
    DOC_TYPES,
    DOC_TYPE_DESCRIPTIONS,
    DOC_TYPE_LABELS,
    find_intent,
    get_intents_for,
    get_prewriting_questions,
)

# ---------------------------------------------------------------------------
# Study writing-task constants
# ---------------------------------------------------------------------------

# Each participant runs a short training session (1 turn) followed by a real
# session (5 turns) on the same writing topic. Training data is flagged with
# `is_training: true` so it can be excluded from analysis.
MIN_EXCHANGES_TRAINING = 1
MIN_EXCHANGES_REAL = 5
DOC_WORD_MIN = 200

# Sampling parameters used for both assistant models at every turn.
GEN_MAX_TOKENS = 2048
GEN_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Pairwise Evaluation UI")

# The study pages are served by this app itself (same origin), so CORS is only
# needed if you embed the study or call the API from another site. Set
# ALLOWED_ORIGINS to a comma-separated list of origins to enable that.
_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Admin authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)
_ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

# ---------------------------------------------------------------------------
# Prolific completion
# ---------------------------------------------------------------------------
_PROLIFIC_COMPLETION_CODE = os.environ.get("PROLIFIC_COMPLETION_CODE", "").strip()
_PROLIFIC_COMPLETION_URL = os.environ.get("PROLIFIC_COMPLETION_URL", "").strip()
if not _PROLIFIC_COMPLETION_URL and _PROLIFIC_COMPLETION_CODE:
    _PROLIFIC_COMPLETION_URL = (
        f"https://app.prolific.com/submissions/complete?cc={_PROLIFIC_COMPLETION_CODE}"
    )


def _require_admin(
    creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
):
    if not _ADMIN_API_KEY:
        return
    if creds is None or creds.credentials != _ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Admin API key required. Set ADMIN_API_KEY env var and pass "
                   "Authorization: Bearer <key> in the request.",
        )


# ---------------------------------------------------------------------------
# Results & static directories
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Study global state
# ---------------------------------------------------------------------------

study_config: Dict = {
    "configured": False,
    # Pool of models. Each entry: {"name": str, "port": int, "client": OpenAI}.
    # Each conversation pairs the two models head-to-head.
    "models": [],
    "min_conversations": 1,
}

_STUDY_CONFIG_PATH = RESULTS_DIR / "_study_config.json"


def _save_study_config() -> None:
    if not study_config.get("configured"):
        return
    snapshot = {
        "models": [
            {"name": m["name"], "port": m["port"]} for m in study_config["models"]
        ],
        "min_conversations": study_config["min_conversations"],
    }
    _STUDY_CONFIG_PATH.write_text(json.dumps(snapshot, indent=2))


def _load_study_config() -> None:
    if not _STUDY_CONFIG_PATH.exists():
        return
    try:
        snapshot = json.loads(_STUDY_CONFIG_PATH.read_text())
        configured = [
            {"name": m["name"], "port": m["port"], "client": get_client(m["name"], m["port"])}
            for m in snapshot["models"]
        ]
        study_config.update({
            "configured": True,
            "models": configured,
            "min_conversations": snapshot.get("min_conversations", 1),
        })
        print(f"[study_config] restored {len(configured)} model(s) from {_STUDY_CONFIG_PATH}")
    except Exception as exc:
        print(f"[study_config] failed to restore from {_STUDY_CONFIG_PATH}: {exc}")


_load_study_config()

participants: Dict[str, Dict] = {}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _moderate(text: str) -> bool:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return True
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.moderations.create(input=text)
        return not resp.results[0].flagged
    except Exception:
        return True


def _get_participant(pid: Optional[str]) -> Optional[Dict]:
    if not pid:
        return None
    return participants.get(pid)


def _writing_setup_ready(setup: Optional[Dict]) -> bool:
    if not setup:
        return False
    return bool(setup.get("doc_type") and setup.get("intent") and setup.get("prewriting_submitted"))


def _next_session_kind(p: Dict) -> Optional[str]:
    """Return the next session kind to run for this participant.

    Order is fixed: one training session, then one real session. Returns None
    once both have been completed (study is done).
    """
    convos = p.get("conversations", [])
    has_training = any(c.get("is_training") for c in convos)
    has_real = any(not c.get("is_training") for c in convos)
    if not has_training:
        return "training"
    if not has_real:
        return "real"
    return None


def _min_exchanges_for(session_kind: str) -> int:
    return MIN_EXCHANGES_TRAINING if session_kind == "training" else MIN_EXCHANGES_REAL


def _snapshot_setup(setup: Optional[Dict]) -> Optional[Dict]:
    if not setup:
        return None
    return {
        "doc_type": setup["doc_type"],
        "intent": setup.get("intent"),
        "prewriting_questions": setup.get("prewriting_questions", []),
        "prewriting_answers": setup.get("prewriting_answers") or [],
    }


def _new_participant(prolific_id: Optional[str] = None) -> str:
    pid = str(uuid.uuid4())
    participants[pid] = {
        "participant_id": pid,
        "prolific_id": prolific_id,
        "created_at": _now(),
        "consent_given": False,
        "consent_timestamp": None,
        # Single writing-task setup (only one task in this variant).
        "writing_setup": None,
        # Pairwise conversations.
        "conversations": [],
        "current": None,
        "completed": False,
        "completion_timestamp": None,
        "reports": [],
        "feedback": [],
    }
    _save_participant(pid)
    return pid


def _init_conversation(pid: str, session_kind: str):
    p = participants[pid]
    models = study_config["models"]
    if len(models) < 2:
        raise HTTPException(
            status_code=400,
            detail="Study must be configured with at least 2 models.",
        )
    # Pick 2 distinct models. With exactly 2 entries this just shuffles them; with more entries it samples 2 without replacement.
    # Re-randomized per /start call so training and real sessions get independent
    # model orderings.
    idx1, idx2 = random.sample(range(len(models)), 2)
    p["current"] = {
        "session_kind": session_kind,
        "is_training": session_kind == "training",
        "model_a_is_model1": random.random() < 0.5,
        "model1_idx": idx1,
        "model2_idx": idx2,
        "model1_name": models[idx1]["name"],
        "model2_name": models[idx2]["name"],
        "hist1": [],
        "hist2": [],
        "turns": [],
        "pending_turn": None,
    }


def _save_participant(pid: str):
    p = participants.get(pid)
    if p is None:
        return
    out_path = RESULTS_DIR / f"{pid}.json"
    safe = {k: v for k, v in p.items()}
    out_path.write_text(json.dumps(safe, indent=2, default=str))


# Secure cookies are required for HTTPS deployments (and for embedding the study
# in an iframe, which needs SameSite=None). For plain-HTTP local testing set
# COOKIE_SECURE=0 so the browser accepts the session cookie.
_COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "1").strip().lower() not in ("0", "false", "no")
_COOKIE_KWARGS = dict(
    httponly=True,
    samesite="none" if _COOKIE_SECURE else "lax",
    secure=_COOKIE_SECURE,
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class StudySessionRequest(BaseModel):
    prolific_id: Optional[str] = None


class StudyConsentRequest(BaseModel):
    agreed: bool
    prolific_id: Optional[str] = None


class TurnRequest(BaseModel):
    query: str


class PreferRequest(BaseModel):
    preference: str  # "model_a" | "model_b" | "tie"
    rationale: str = ""


class WritingIntentRequest(BaseModel):
    intent_id: str


class WritingPrewritingRequest(BaseModel):
    answers: List[str]


class ReportRequest(BaseModel):
    content: str
    turn_idx: Optional[int] = None


class FeedbackRequest(BaseModel):
    content: str


class FinishRequest(BaseModel):
    final_document: str = ""


class ModelSpec(BaseModel):
    name: str
    port: int


class AdminStudyConfigRequest(BaseModel):
    # Must contain exactly the two models you want to compare. Each conversation
    # pairs them head-to-head.
    models: List[ModelSpec]
    min_conversations: int = 1


# ---------------------------------------------------------------------------
# Study page routes
# ---------------------------------------------------------------------------

@app.get("/study")
def study_consent_page():
    return FileResponse(_static_dir / "consent.html")


@app.get("/study/instructions")
def study_instructions_page():
    return FileResponse(_static_dir / "instructions.html")


@app.get("/study/instructions2")
def study_instructions2_page():
    return FileResponse(_static_dir / "instructions2.html")


@app.get("/study/setup")
def study_setup_page():
    return FileResponse(_static_dir / "setup.html")


@app.get("/study/task")
def study_task_page():
    return FileResponse(_static_dir / "task.html")


@app.get("/study/done")
def study_done_page():
    return FileResponse(_static_dir / "done.html")


@app.get("/study/declined")
def study_declined_page():
    return FileResponse(_static_dir / "declined.html")


# ---------------------------------------------------------------------------
# Study API endpoints  (public — cookie-authenticated per-participant)
# ---------------------------------------------------------------------------

@app.post("/api/study/session")
def study_session(
    req: StudySessionRequest,
    response: Response,
    pid: Optional[str] = Cookie(None),
):
    p = _get_participant(pid)
    if p is None:
        pid = _new_participant(prolific_id=req.prolific_id)
        p = participants[pid]

    response.set_cookie(key="pid", value=pid, **_COOKIE_KWARGS)
    return {
        "participant_id": pid,
        "consent_given": p["consent_given"],
        "conversations_done": len(p["conversations"]),
        "min_conversations": study_config["min_conversations"],
        "study_configured": study_config["configured"],
    }


@app.post("/api/study/consent")
def study_consent(req: StudyConsentRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session found. Please start from /study.")

    prolific_id = (req.prolific_id or "").strip()
    if req.agreed and not prolific_id and not p.get("prolific_id"):
        raise HTTPException(status_code=400, detail="Prolific ID is required to give consent.")

    p["consent_given"] = req.agreed
    p["consent_timestamp"] = _now()
    if prolific_id:
        p["prolific_id"] = prolific_id
    _save_participant(pid)

    return {"ok": True, "agreed": req.agreed}


def _writing_setup_assign(p: Dict) -> Dict:
    doc_type = random.choice(DOC_TYPES)
    intents = get_intents_for(doc_type)
    questions = get_prewriting_questions(doc_type)
    setup = {
        "doc_type": doc_type,
        "intent": None,
        "prewriting_questions": questions,
        "prewriting_answers": None,
        "prewriting_submitted": False,
        "assigned_at": _now(),
        "submitted_at": None,
    }
    p["writing_setup"] = setup
    return {
        "doc_type": doc_type,
        "doc_type_label": DOC_TYPE_LABELS[doc_type],
        "doc_type_description": DOC_TYPE_DESCRIPTIONS[doc_type],
        "intents": intents,
        "prewriting_questions": questions,
    }


def _writing_setup_info(p: Dict) -> Optional[Dict]:
    setup = p.get("writing_setup")
    if not setup:
        return None
    return {
        "doc_type": setup["doc_type"],
        "doc_type_label": DOC_TYPE_LABELS[setup["doc_type"]],
        "doc_type_description": DOC_TYPE_DESCRIPTIONS[setup["doc_type"]],
        "intents": get_intents_for(setup["doc_type"]),
        "intent": setup.get("intent"),
        "prewriting_questions": setup.get("prewriting_questions", []),
        "prewriting_answers": setup.get("prewriting_answers") or [],
        "prewriting_submitted": bool(setup.get("prewriting_submitted")),
    }


@app.post("/api/study/setup/assign")
def setup_assign(pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")
    existing = p.get("writing_setup")
    if _writing_setup_ready(existing):
        return _writing_setup_info(p)
    info = _writing_setup_assign(p)
    _save_participant(pid)
    return info


@app.post("/api/study/setup/intent")
def setup_intent(req: WritingIntentRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")
    setup = p.get("writing_setup")
    if not setup:
        raise HTTPException(status_code=400, detail="No writing setup assigned.")
    intent = find_intent(setup["doc_type"], req.intent_id)
    if intent is None:
        raise HTTPException(status_code=400, detail="Unknown intent id for this doc type.")
    setup["intent"] = intent
    _save_participant(pid)
    return {"ok": True, "intent": intent}


@app.post("/api/study/setup/prewriting")
def setup_prewriting(req: WritingPrewritingRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")
    setup = p.get("writing_setup")
    if not setup or not setup.get("intent"):
        raise HTTPException(status_code=400, detail="Select an intent before submitting pre-writing notes.")
    questions = setup.get("prewriting_questions", [])
    if len(req.answers) != len(questions):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(questions)} answers, got {len(req.answers)}.",
        )
    setup["prewriting_answers"] = [a.strip() for a in req.answers]
    setup["prewriting_submitted"] = True
    setup["submitted_at"] = _now()
    _save_participant(pid)
    return {"ok": True}


@app.get("/api/study/setup")
def setup_get(pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    info = _writing_setup_info(p)
    if info is None:
        return {"assigned": False}
    return {"assigned": True, **info}


@app.post("/api/study/start")
def start(pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")
    if not study_config["configured"]:
        raise HTTPException(status_code=400, detail="Study not configured by admin yet.")
    if not _writing_setup_ready(p.get("writing_setup")):
        raise HTTPException(
            status_code=400,
            detail="Complete the writing setup (intent + pre-writing) before starting the task.",
        )

    session_kind = _next_session_kind(p)
    if session_kind is None:
        raise HTTPException(status_code=400, detail="Study already complete.")

    # If a previous conversation was in progress and never finished (e.g. the
    # participant refreshed the page), discard it silently and start fresh.
    _init_conversation(pid, session_kind=session_kind)
    _save_participant(pid)

    setup = p["writing_setup"]
    return {
        "ok": True,
        "conversation_idx": len(p["conversations"]),
        "session_kind": session_kind,
        "is_training": session_kind == "training",
        "intent": setup["intent"],
        "doc_type": setup["doc_type"],
        "doc_type_label": DOC_TYPE_LABELS[setup["doc_type"]],
        "prewriting_questions": setup["prewriting_questions"],
        "prewriting_answers": setup["prewriting_answers"],
        "min_exchanges": _min_exchanges_for(session_kind),
        "doc_word_min": DOC_WORD_MIN,
    }


@app.post("/api/study/turn")
def turn(req: TurnRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")
    if not study_config["configured"]:
        raise HTTPException(status_code=400, detail="Study not configured.")

    current = p.get("current")
    if current is None:
        raise HTTPException(status_code=400, detail="No active conversation. Call /api/study/start first.")
    if current.get("pending_turn") is not None:
        raise HTTPException(status_code=400, detail="Please select a preference for the current turn first.")

    query_at = _now()
    models = study_config["models"]
    m1 = models[current["model1_idx"]]
    m2 = models[current["model2_idx"]]

    # No system prompt — both AIs see only the conversation.
    r1, r2 = generate_pair(
        m1["client"], m1["name"], current["hist1"],
        m2["client"], m2["name"], current["hist2"],
        req.query,
        GEN_MAX_TOKENS, GEN_TEMPERATURE, None,
        is_allowed=_moderate,
    )
    responded_at = _now()

    # Randomize left/right display independently each turn to reduce position bias.
    a_is_1 = random.random() < 0.5
    ra = r1 if a_is_1 else r2
    rb = r2 if a_is_1 else r1

    turn_idx = len(current["turns"])
    pending = {
        "turn_idx": turn_idx,
        "user_query": req.query,
        "model1_response": r1,
        "model2_response": r2,
        "model_a_is_model1": a_is_1,
        "response_a": ra,
        "response_b": rb,
        "preference": None,
        "query_at": query_at,
        "responded_at": responded_at,
        "preferred_at": None,
    }
    current["pending_turn"] = pending
    _save_participant(pid)

    return {"turn_idx": turn_idx, "user_query": req.query, "response_a": ra, "response_b": rb}


@app.post("/api/study/prefer")
def prefer(req: PreferRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")

    current = p.get("current")
    if current is None:
        raise HTTPException(status_code=400, detail="No active conversation.")
    if current.get("pending_turn") is None:
        raise HTTPException(status_code=400, detail="No pending turn.")
    if req.preference not in ("model_a", "model_b", "tie"):
        raise HTTPException(status_code=400, detail="Invalid preference.")

    pending = current["pending_turn"]
    query = pending["user_query"]
    r1 = pending["model1_response"]
    r2 = pending["model2_response"]

    a_is_1 = pending["model_a_is_model1"]
    if req.preference == "model_a":
        pref_internal = "model1" if a_is_1 else "model2"
    elif req.preference == "model_b":
        pref_internal = "model2" if a_is_1 else "model1"
    else:
        pref_internal = "tie"

    pending["preference"] = pref_internal
    pending["preference_ab"] = req.preference
    pending["rationale"] = req.rationale
    pending["preferred_at"] = _now()

    if pref_internal == "model1":
        preferred_response = r1
    elif pref_internal == "model2":
        preferred_response = r2
    else:
        preferred_response = pending["response_a"]

    shared_turn = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": preferred_response},
    ]
    current["hist1"].extend(shared_turn)
    current["hist2"].extend(shared_turn)

    current["turns"].append(pending)
    current["pending_turn"] = None
    _save_participant(pid)

    m1_name = current["model1_name"]
    m2_name = current["model2_name"]
    model_a_name = m1_name if a_is_1 else m2_name
    model_b_name = m2_name if a_is_1 else m1_name

    return {
        "ok": True,
        "turn_idx": pending["turn_idx"],
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
    }


@app.post("/api/study/finish")
def finish(req: FinishRequest, pid: Optional[str] = Cookie(None)):
    """Finish the current pairwise conversation.

    For the real session, the participant must paste their final document
    (>= DOC_WORD_MIN words). The training session auto-ends after one
    exchange and does not collect a document.
    """
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")

    current = p.get("current")
    if current is None:
        raise HTTPException(status_code=400, detail="No active conversation.")
    session_kind = current.get("session_kind", "real")
    min_required_turns = _min_exchanges_for(session_kind)
    if len(current.get("turns", [])) < min_required_turns:
        raise HTTPException(
            status_code=400,
            detail=f"You need at least {min_required_turns} exchanges before finishing this conversation.",
        )

    final_document = req.final_document.strip()
    if session_kind != "training":
        word_count = len(final_document.split())
        if word_count < DOC_WORD_MIN:
            raise HTTPException(
                status_code=400,
                detail=f"Final document must be at least {DOC_WORD_MIN} words (got {word_count}).",
            )

    p["conversations"].append({
        "session_kind": session_kind,
        "is_training": session_kind == "training",
        "model_a_is_model1": current["model_a_is_model1"],
        "model1_name": current["model1_name"],
        "model2_name": current["model2_name"],
        "setup": _snapshot_setup(p.get("writing_setup")),
        "turns": current.get("turns", []),
        "final_document": final_document,
        "finished_at": _now(),
    })
    p["current"] = None

    next_kind = _next_session_kind(p)
    study_complete = next_kind is None

    _save_participant(pid)
    return {
        "ok": True,
        "session_kind": session_kind,
        "next_session_kind": next_kind,
        "study_complete": study_complete,
    }


@app.post("/api/study/report")
def study_report(req: ReportRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")

    p["reports"].append({
        "content": req.content,
        "turn_idx": req.turn_idx,
        "timestamp": _now(),
    })
    _save_participant(pid)
    return {"ok": True}


@app.post("/api/study/feedback")
def study_feedback(req: FeedbackRequest, pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")

    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Feedback cannot be empty.")

    p.setdefault("feedback", []).append({
        "content": content,
        "timestamp": _now(),
    })
    _save_participant(pid)
    return {"ok": True}


@app.post("/api/study/complete")
def study_complete(pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        raise HTTPException(status_code=401, detail="No session.")
    if not p["consent_given"]:
        raise HTTPException(status_code=403, detail="Consent required.")

    completion_code = _PROLIFIC_COMPLETION_CODE
    completion_url = _PROLIFIC_COMPLETION_URL

    if p.get("completed"):
        return {
            "ok": True,
            "completion_code": completion_code,
            "completion_url": completion_url,
        }

    if _next_session_kind(p) is not None:
        raise HTTPException(
            status_code=400,
            detail="You need to complete both the training and real sessions before finishing the study.",
        )

    p["completed"] = True
    p["completion_timestamp"] = _now()
    p["completion_code"] = completion_code
    _save_participant(pid)

    return {
        "ok": True,
        "completion_code": completion_code,
        "completion_url": completion_url,
    }


@app.get("/api/study/status")
def study_status(pid: Optional[str] = Cookie(None)):
    p = _get_participant(pid)
    if p is None:
        return {
            "has_session": False,
            "consent_given": False,
            "setup_ready": False,
            "next_session_kind": "training",
            "training_done": False,
            "completed": False,
            "completion_code": "",
            "completion_url": "",
            "study_configured": study_config["configured"],
        }

    next_kind = _next_session_kind(p)
    return {
        "has_session": True,
        "consent_given": p["consent_given"],
        "setup_ready": _writing_setup_ready(p.get("writing_setup")),
        "next_session_kind": next_kind,
        "training_done": next_kind != "training",
        "completed": p["completed"],
        "completion_code": _PROLIFIC_COMPLETION_CODE if p["completed"] else "",
        "completion_url": _PROLIFIC_COMPLETION_URL if p["completed"] else "",
        "study_configured": study_config["configured"],
    }


# ---------------------------------------------------------------------------
# Admin endpoints  (require ADMIN_API_KEY)
# ---------------------------------------------------------------------------

@app.post("/api/admin/study-config")
def admin_study_config(req: AdminStudyConfigRequest, _: None = Depends(_require_admin)):
    if len(req.models) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models.")
    names = [m.name for m in req.models]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Duplicate model names in pool.")
    try:
        configured = [
            {"name": m.name, "port": m.port, "client": get_client(m.name, m.port)}
            for m in req.models
        ]
        study_config.update({
            "configured": True,
            "models": configured,
            "min_conversations": req.min_conversations,
        })
        _save_study_config()
        return {"ok": True, "n_models": len(configured), "model_names": names}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/results")
def admin_results(_: None = Depends(_require_admin)):
    total = len(participants)
    consented = sum(1 for p in participants.values() if p["consent_given"])
    convo_total = completed = 0

    for p in participants.values():
        convo_total += len(p["conversations"])
        if p["completed"]:
            completed += 1

    return {
        "total_participants": total,
        "consented": consented,
        "conversations_total": convo_total,
        "completed": completed,
    }


# ---------------------------------------------------------------------------
# Static file serving — must come last
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
