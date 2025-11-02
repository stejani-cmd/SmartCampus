import re
from fastapi import Request
from datetime import datetime
from fastapi import UploadFile, File, HTTPException
from fastapi import HTTPException, Request
import anyio
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from pymongo import MongoClient
import gridfs
import os
import json
from pathlib import Path
from fastapi.responses import JSONResponse
from datetime import date, datetime, timedelta
from bson import ObjectId
import io
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, ValidationError
import logging
from app.utils import get_current_user
from app.routers import assignment_checker
from app.routers import assignment_checker

from app.routers import auth, pages, forum
from app.core.config import settings
import os
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- env ----------
load_dotenv()
print("[BOOT] USE_LLM_FOLLOWUPS=", os.getenv("USE_LLM_FOLLOWUPS", "1"),
      "MODEL=", os.getenv("FOLLOWUP_MODEL", "gpt-4o-mini"),
      "OPENAI_KEY_PRESENT=", bool(os.getenv("OPENAI_API_KEY")))

# ------------------ FastAPI App ------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")  # folder for HTML templates

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY"),
    same_site="lax",   # or "none" if using HTTPS
    https_only=False,
    max_age=3600  # Session expires after 1 hour
)



UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")



import fitz  # PyMuPDF

def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts)


# routers
app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(forum.router)
app.include_router(assignment_checker.router)

# Configure OAuth for Google
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# ------------------ MongoDB Setup ------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017/smartassist")
client = MongoClient(MONGO_URI)
db = client.smartassist
users_collection = db.users
live_chat_collection = db.live_chat
live_chat_sessions = db.live_chat_sessions
fs = gridfs.GridFS(db)

# Ensure a text index exists for follow-ups (safe to call once)
try:
    db.articles.create_index(
        [("title", "text"), ("content", "text"), ("category", "text")])
except Exception:
    pass

# ======================================================================================
#                                    OpenAI SHIM
# ======================================================================================
def llm_complete(
    messages,
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=400,  # Increased default
    response_format: dict | None = None  # 👈 ADDED THIS
) -> str:
    """
    Returns assistant text using whichever OpenAI SDK is installed.
    Uses a legacy-safe model when v0 SDK is detected.
    """
    # Try v1 SDK
    try:
        from openai import OpenAI  # v1+
        client = OpenAI()

        # 👇 ADDED THIS LOGIC
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            params["response_format"] = response_format
        # 👆 END ADDED LOGIC

        resp = client.chat.completions.create(**params) # 👈 UPDATED THIS
        return resp.choices[0].message.content.strip()

    except Exception as v1_err:
        # Fallback to legacy v0 SDK
        import openai
        if not getattr(openai, "api_key", None):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        legacy_model = os.getenv("FOLLOWUP_MODEL_LEGACY", "gpt-3.5-turbo")
        
        # Note: v0 SDK does not support response_format, so we can't use it here.
        if response_format:
            raise RuntimeError(f"JSON mode (response_format) is not supported by legacy v0 SDK. {v1_err!r}")
            
        try:
            resp = openai.ChatCompletion.create(
                model=legacy_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as v0_err:
            # bubble up a unified error so caller can switch to fallback
            raise RuntimeError(
                f"OpenAI failed (v1: {v1_err!r}; v0: {v0_err!r})")


# ======================================================================================
#                        LLM-STYLE, INTENT-LIKE FOLLOW-UPS
# ======================================================================================
USE_LLM_FOLLOWUPS = os.getenv("USE_LLM_FOLLOWUPS", "1") == "1"
FOLLOWUP_MODEL = os.getenv("FOLLOWUP_MODEL", "gpt-4o-mini")
DEBUG_FOLLOWUPS = os.getenv("DEBUG_FOLLOWUPS", "1") == "1"

ESCALATION_KEYWORDS = {
    "agent", "human", "person", "representative", "talk to someone", "talk to admin",
    "live chat", "connect me", "escalate", "call", "phone", "help desk", "support"
}


def _wants_human(text: str) -> bool:
    q = (text or "").lower()
    return any(k in q for k in ESCALATION_KEYWORDS)


def _mongo_text_search(query: str, limit: int = 8) -> List[Dict]:
    if not (query and query.strip()):
        return []
    cur = (db.articles.find(
        {"$text": {"$search": query}},
        {"title": 1, "category": 1, "url": 1, "score": {"$meta": "textScore"}}
    )
        .sort([("score", {"$meta": "textScore"})])
        .limit(limit))
    return list(cur)


def _should_offer_live_chat(user_q: str, answer_text: str, hits: int) -> bool:
    if _wants_human(user_q):
        return True
    low_conf = [
        "i'm not sure", "no information", "could not find", "not available",
        "i don't have", "unable to find"
    ]
    a = (answer_text or "").lower()
    return hits == 0 or any(p in a for p in low_conf)


def _safe_json_list(s: str) -> List[str]:
    """
    Be forgiving: extract the first JSON array from the text and parse it.
    Returns [] if nothing usable is found.
    """
    if not s:
        return []
    # already a pure array?
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, (str,))]
    except Exception:
        pass

    # try to find a [...] substring
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        frag = m.group(0)
        try:
            data = json.loads(frag)
            if isinstance(data, list):
                return [str(x) for x in data if isinstance(x, (str,))]
        except Exception:
            return []
    return []


def _llm_generate_followups(user_q: str, answer_text: str, candidates: List[Dict], k: int = 4) -> List[str]:
    # pack candidate KB as lightweight grounding
    ctx_lines = []
    for c in candidates[:10]:
        t = (c.get("title") or "").strip()
        u = c.get("url", "")
        cat = c.get("category", "")
        if t:
            ctx_lines.append(f"- {t} [{cat}] {u}")
    ctx = "\n".join(ctx_lines) if ctx_lines else "(no candidates)"

    sys = (
        "You are a campus assistant crafting follow-up suggestions that feel like the student's next question. "
        "Return ONLY a JSON array of 3-5 strings. Constraints: "
        "1) Each suggestion must be a natural, concise user question (max ~10 words). "
        "2) Avoid vague labels and categories; be specific. "
        "3) No duplicates, no punctuation at the end, no numbering. "
        "4) Suggestions must be answerable by the provided knowledge items when possible."
    )
    usr = (
        f"Student question: {user_q}\n\n"
        f"Assistant answer:\n{answer_text}\n\n"
        f"Relevant knowledge items:\n{ctx}\n\n"
        "Produce a JSON array of short follow-up questions likely to be asked next."
    )

    text = llm_complete(
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": usr}],
        model=FOLLOWUP_MODEL,
        temperature=0.4,
        max_tokens=180,
    )
    items = _safe_json_list(text)
    uniq, seen = [], set()
    for it in items:
        s = it.strip()
        if s.endswith("?"):
            s = s[:-1]
        if s and s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
        if len(uniq) >= k:
            break
    return uniq


def build_llm_style_followups(user_question: str, answer_text: str, k: int = 4):
    """
    Returns (chips, suggest_live_chat_flag, source).
    source ∈ {"openai","fallback","fallback_error"} to help you verify.
    """
    hits = _mongo_text_search(user_question, limit=8)
    if not hits and answer_text:
        hits = _mongo_text_search(answer_text, limit=8)

    suggestions: List[str] = []
    source = "fallback"

    if USE_LLM_FOLLOWUPS and os.getenv("OPENAI_API_KEY"):
        try:
            suggestions = _llm_generate_followups(
                user_question, answer_text, hits, k=k)
            if suggestions:
                source = "openai"
        except Exception as e:
            print("[LLM] followups error:", repr(e))
            suggestions = []
            source = "fallback_error"

    if not suggestions:
        # graceful fallback: try KB titles, then a tiny curated list
        base = [h.get("title", "") for h in hits[:6] if h.get("title")]
        suggestions = [s for s in base if s][:k]
        if not suggestions:
            suggestions = [
                "application deadlines for scholarships",
                "GPA needed for freshman admission",
                "who is my admissions counselor",
                "how to apply for scholarships",
            ][:k]

    chips = [{"label": s, "payload": {"type": "faq", "query": s}}
             for s in suggestions]
    suggest_live_chat = _should_offer_live_chat(
        user_question, answer_text, hits=len(hits))
    return chips[:k], suggest_live_chat, source


# ======================================================================================
#                               LLM DIAGNOSTIC ENDPOINT
# ======================================================================================
@app.get("/diag/llm")
def diag_llm():
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=500, detail="OPENAI_API_KEY missing")
        text = llm_complete(
            messages=[{"role": "system", "content": "Return the word OK"}],
            model=FOLLOWUP_MODEL,
            temperature=0.0,
            max_tokens=4,
        )
        return {"ok": True, "model": FOLLOWUP_MODEL, "reply": text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


print("[BOOT]",
      "USE_LLM_FOLLOWUPS=", USE_LLM_FOLLOWUPS,
      "MODEL=", FOLLOWUP_MODEL,
      "OPENAI_KEY_PRESENT=", bool(os.getenv("OPENAI_API_KEY")))

# ======================================================================================
#                                 LIVE CHAT MANAGER
# ======================================================================================


class ChatManager:
    def __init__(self):
        self.admins: List[WebSocket] = []
        self.students: dict = {}  # session_id -> WebSocket

    async def connect_admin(self, websocket: WebSocket):
        await websocket.accept()
        self.admins.append(websocket)
        print("✅ Admin connected")

    async def connect_student(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.students[session_id] = websocket
        print(f"✅ Student connected: {session_id}")

        # ensure record; DO NOT broadcast yet (only /escalate does)
        live_chat_sessions.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id": session_id,
                    "status": "queued",
                    "assigned_admin": None,
                    "name": f"Student {session_id[:4]}"
                },
                "$set": {"student_connected": True}
            },
            upsert=True
        )

        if not live_chat_collection.find_one({"session_id": session_id}):
            live_chat_collection.insert_one({
                "session_id": session_id,
                "sender": "system",
                "message": "New chat session started.",
                "created_at": datetime.utcnow()
            })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.admins:
            self.admins.remove(websocket)
            print("❌ Admin disconnected")
        else:
            for sid, ws in list(self.students.items()):
                if ws == websocket:
                    del self.students[sid]
                    live_chat_sessions.update_one(
                        {"session_id": sid},
                        {"$set": {"student_connected": False, "status": "closed"}}
                    )
                    # remove from admin UI immediately
                    anyio.from_thread.run(self.broadcast_admins, {
                                          "type": "session_removed", "session_id": sid})
                    print(f"❌ Student disconnected: {sid}")

    async def send_to_student(self, session_id: str, message: dict):
        if session_id in self.students:
            await self.students[session_id].send_json(message)

    async def broadcast_admins(self, message: dict):
        for admin in list(self.admins):
            try:
                await admin.send_json(message)
            except Exception:
                # stale socket
                pass


manager = ChatManager()


def save_ticket(ticket: dict, attachment: UploadFile | None = None):
    # store ticket in MongoDB; if attachment provided, save in GridFS and reference file id
    if attachment is not None:
        try:
            content = attachment.file.read()
            file_id = fs.put(content, filename=attachment.filename,
                             contentType=attachment.content_type)
            ticket["attachment_id"] = file_id
            ticket["attachment_name"] = attachment.filename
            ticket["attachment_content_type"] = attachment.content_type
        except Exception as e:
            ticket["attachment_error"] = f"failed to save to gridfs: {e}"

    result = db.tickets.insert_one(ticket)
    inserted_id = result.inserted_id

    # Debug output: print inserted id and ticket document (attachment_id as string)
    debug_doc = ticket.copy()
    if "attachment_id" in debug_doc:
        debug_doc["attachment_id"] = str(debug_doc["attachment_id"])
    try:
        print(f"[DEBUG] Inserted ticket id: {inserted_id}")
        print(f"[DEBUG] Ticket document: {json.dumps(debug_doc, default=str)}")
    except Exception:
        # fallback simple print if json.dumps fails
        print("[DEBUG] Ticket document (fallback):", debug_doc)

    return inserted_id


@app.post("/raise_ticket")
async def raise_ticket(
    subject: str = Form(...),
    category: str = Form(...),
    priority: str = Form(...),
    description: str = Form(...),
    attachment: UploadFile | None = File(None)
):
    # Basic validation
    if not subject or not category or not priority or not description:
        return JSONResponse({"success": False, "error": "Missing required fields"}, status_code=400)

    ticket = {
        "subject": subject,
        "category": category,
        "priority": priority,
        "description": description,
        "status": "open"
    }

    try:
        inserted_id = save_ticket(ticket, attachment)
        print(
            f"[DEBUG] /raise_ticket: inserted_id={inserted_id} attachment_present={attachment is not None}")
        return {"success": True, "ticket_id": str(inserted_id)}
    except Exception as e:
        print(f"[ERROR] /raise_ticket exception: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


def save_appointment(appt: dict, attachment: UploadFile | None = None):
    # store appointment in MongoDB; if attachment provided, save in GridFS and reference file id
    if attachment is not None:
        try:
            content = attachment.file.read()
            file_id = fs.put(content, filename=attachment.filename,
                             contentType=attachment.content_type)
            appt["attachment_id"] = file_id
            appt["attachment_name"] = attachment.filename
            appt["attachment_content_type"] = attachment.content_type
        except Exception as e:
            appt["attachment_error"] = f"failed to save to gridfs: {e}"

    result = db.appointments.insert_one(appt)
    inserted_id = result.inserted_id

    # Debug output: print inserted id and appointment document (attachment_id as string)
    debug_doc = appt.copy()
    if "attachment_id" in debug_doc:
        debug_doc["attachment_id"] = str(debug_doc["attachment_id"])
    try:
        print(f"[DEBUG] Inserted appointment id: {inserted_id}")
        print(
            f"[DEBUG] Appointment document: {json.dumps(debug_doc, default=str)}")
    except Exception:
        print("[DEBUG] Appointment document (fallback):", debug_doc)

    return inserted_id


# ======================================================================================
#                                   WEBSOCKETS
# ======================================================================================
@app.websocket("/ws/student/{session_id}")
async def student_ws(websocket: WebSocket, session_id: str):
    print(f"[DEBUG] Student connected with session_id: {session_id}")
    await manager.connect_student(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_json()
            message_text = data.get("message", "")
            print(f"[DEBUG] Received message from student: {message_text}")

            live_chat_collection.insert_one({
                "session_id": session_id,
                "sender": "student",
                "message": message_text,
                "created_at": datetime.utcnow()
            })

            sess = live_chat_sessions.find_one({"session_id": session_id})
            if sess and sess.get("status") == "live":
                await manager.broadcast_admins({
                    "type": "message",
                    "session_id": session_id,
                    "sender": "student",
                    "message": message_text
                })
            else:
                # Calculate queue position
                queued_sessions = list(live_chat_sessions.find(
                    {"status": "queued"}).sort("created_at", 1))
                queue_position = next(
                    (i + 1 for i, s in enumerate(queued_sessions) if s["session_id"] == session_id), None)

                await manager.broadcast_admins({
                    "type": "queued_ping",
                    "session_id": session_id,
                    "queue_position": queue_position
                })

    except WebSocketDisconnect:
        print(f"[DEBUG] Student disconnected with session_id: {session_id}")
        manager.disconnect(websocket)


@app.websocket("/ws/admin")
async def admin_ws(websocket: WebSocket):
    print("[DEBUG] /ws/admin endpoint accessed")
    print("[DEBUG] Admin connected")
    await manager.connect_admin(websocket)
    admin_id = str(id(websocket))

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            print(f"[DEBUG] Received message from admin: {data}")

            if msg_type == "join":
                session_id = data.get("session_id")
                sess = live_chat_sessions.find_one({"session_id": session_id})
                if not sess or not sess.get("student_connected") or sess.get("status") == "closed":
                    await websocket.send_json({"type": "error", "reason": "Student not connected / session closed."})
                    await websocket.send_json({"type": "session_removed", "session_id": session_id})
                    continue

                res = live_chat_sessions.update_one(
                    {"session_id": session_id, "status": {
                        "$in": ["queued", "live"]}},
                    {"$set": {"status": "live", "assigned_admin": admin_id}}
                )
                if res.matched_count == 0:
                    await websocket.send_json({"type": "error", "reason": "Session not found or closed."})
                    continue

                await manager.send_to_student(session_id, {
                    "type": "status",
                    "session_id": session_id,
                    "status": "live"
                })
                await websocket.send_json({"type": "joined", "session_id": session_id})

            elif msg_type == "message":
                session_id = data.get("session_id")
                message_text = data.get("message", "")

                sess = live_chat_sessions.find_one({"session_id": session_id})
                if not sess or sess.get("status") != "live" or sess.get("assigned_admin") != admin_id:
                    await websocket.send_json({"type": "error", "reason": "Session not live or not assigned to you."})
                    continue

                live_chat_collection.insert_one({
                    "session_id": session_id,
                    "sender": "admin",
                    "message": message_text,
                    "created_at": datetime.utcnow()
                })
                await manager.send_to_student(session_id, {
                    "type": "message",
                    "session_id": session_id,
                    "sender": "admin",
                    "message": message_text
                })

            else:
                await websocket.send_json({"type": "error", "reason": "Unknown message type."})

    except WebSocketDisconnect:
        print("[DEBUG] Admin disconnected")
        manager.disconnect(websocket)

# ======================================================================================
#                                   REST API
# ======================================================================================


@app.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    print(f"[DEBUG] Fetching chat history for session_id: {session_id}")
    messages = list(live_chat_collection
                    .find({"session_id": session_id}, {"_id": 0})
                    .sort("created_at", 1))
    print(f"[DEBUG] Retrieved messages: {messages}")
    return messages


@app.post("/api/chat/{session_id}/escalate")
async def escalate(session_id: str):
    # student has asked for an agent — surface to admins now
    live_chat_sessions.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {"session_id": session_id, "assigned_admin": None},
            "$set": {"status": "queued", "student_connected": True, "name": f"Student {session_id[:4]}"},
        },
        upsert=True
    )
    await manager.broadcast_admins({
        "type": "new_session",
        "session_id": session_id,
        "status": "queued",
        "name": f"Student {session_id[:4]}"
    })
    return {"ok": True}


@app.post("/api/chat/{session_id}/end")
async def end_chat(session_id: str):
    live_chat_sessions.update_one(
        {"session_id": session_id},
        {"$set": {
            "status": "closed",
            "student_connected": False,
            "assigned_admin": None,
            "ended_at": datetime.utcnow()
        }}
    )
    await manager.broadcast_admins({"type": "session_removed", "session_id": session_id})
    return {"ok": True}


@app.get("/api/admin/live_chats")
async def list_live_chats():
    docs = list(live_chat_sessions.find({}, {"_id": 0}))
    for d in docs:
        if not d.get("name"):
            sid = d.get("session_id", "")
            d["name"] = f"Student {sid[:4]}" if sid else "Student"
    order = {"queued": 0, "live": 1, "closed": 2}
    docs.sort(key=lambda x: order.get(x.get("status", "queued"), 9))
    return docs

# ======================================================================================
#                               PAGES & AUTH
# ======================================================================================


@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# ---- Register (GET + POST) ----


@app.get("/register", response_class=HTMLResponse)
async def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def post_register(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Passwords do not match!"}
        )

    if users_collection.find_one({"email": email}):
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Email already registered!"}
        )

    users_collection.insert_one({
        "full_name": full_name,
        "email": email,
        "password": password,       # TODO: hash this
        "role": "student"           # default role
    })
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": "Registration successful! Please login."}
    )


""" @app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request}) """


@app.post("/login")
async def post_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    user = users_collection.find_one({"email": email})
    if user and user["password"] == password and user["role"] == role:
        request.session["user"] = {
            "full_name": user["full_name"],
            "email": user["email"],
            "role": user["role"]
        }
        # Debug log to confirm session set
        print("[DEBUG] Session set for user:", request.session["user"])
        if role == "student":
            return RedirectResponse("/student_home", status_code=302)
        elif role == "staff":
            return RedirectResponse("/staff_home", status_code=302)
        elif role == "admin":
            return RedirectResponse("/admin_home", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials or role!"})

# Add session validation and role-based access control to protect dashboard routes


# Middleware to validate session and restrict access based on roles
def get_current_user(request: Request):
    user = request.session.get("user")
    print("[DEBUG] Current session user:", user)  # Debug log
    if not user:
        request.session.clear()  # Ensure session is cleared if no user
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not user.get("role") or not user.get("email"):
        request.session.clear()  # Clear stale session
        print("[DEBUG] Stale session cleared")  # Debug log
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

# get mail


def get_student_email_from_request(request: Request, fallback: str | None = None):
    # 1) if frontend sent the email, use that
    if fallback:
        return fallback
    # 2) else try session.user.email
    user = request.session.get("user")
    if user and user.get("email"):
        return user["email"]
    return None


def role_required(required_role: str):
    def role_dependency(user: dict = Depends(get_current_user)):
        if user.get("role") != required_role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return role_dependency


@app.get("/student_home", response_class=HTMLResponse)
async def student_dashboard(request: Request, user: dict = Depends(role_required("student"))):
    return templates.TemplateResponse("student_home.html", {"request": request})


@app.get("/staff_home", response_class=HTMLResponse)
async def staff_dashboard(request: Request, user: dict = Depends(role_required("staff"))):
    return templates.TemplateResponse("staff_home.html", {"request": request})


@app.get("/admin_home", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: dict = Depends(role_required("admin"))):
    return templates.TemplateResponse("admin_home.html", {"request": request})


@app.get("/knowledge_base", response_class=HTMLResponse)
async def knowledge_base(request: Request, user: dict = Depends(role_required("admin"))):
    return templates.TemplateResponse("knowledge_base.html", {"request": request})


@app.get("/edit_profile", response_class=HTMLResponse)
async def edit_profile(request: Request, user: dict = Depends(role_required("student"))):
    return templates.TemplateResponse("edit_profile.html", {"request": request})


@app.get("/guest_home", response_class=HTMLResponse)
async def guest_dashboard(request: Request, user: dict = Depends(role_required("guest"))):
    return templates.TemplateResponse("guest_home.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: dict = Depends(get_current_user)):
    if user.get("role") not in ["guest", "student", "admin"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    user = request.session.get("user")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": user,   # 👈 so Jinja can do {{ user.email }}
        },
    )


# ------------------ Ticket & Appointment Endpoints ------------------

@app.post("/book_appointment")
async def book_appointment(
    type: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    notes: str = Form(""),
    attachment: UploadFile | None = File(None)
):
    if not type or not date or not time:
        return JSONResponse({"success": False, "error": "Missing required fields"}, status_code=400)

    appt = {
        "type": type,
        "date": date,
        "time": time,
        "notes": notes,
        "status": "scheduled"
    }

    try:
        inserted_id = save_appointment(appt, attachment)
        print(
            f"[DEBUG] /book_appointment: inserted_id={inserted_id} attachment_present={attachment is not None}")
        return {"success": True, "appointment_id": str(inserted_id)}
    except Exception as e:
        print(f"[ERROR] /book_appointment exception: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# Cancel a ticket


@app.post("/api/tickets/cancel/{ticket_id}")
async def cancel_ticket(ticket_id: str):
    try:
        result = db.tickets.update_one(
            {"_id": ObjectId(ticket_id)},
            {"$set": {"status": "Cancelled", "last_updated": datetime.now().isoformat()}}
        )
        if result.modified_count == 1:
            return {"success": True, "message": "Ticket cancelled successfully."}
        return {"success": False, "message": "Ticket not found or already cancelled."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Cancel an appointment


@app.post("/api/appointments/cancel/{appointment_id}")
async def cancel_appointment(appointment_id: str):
    try:
        result = db.appointments.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {"status": "Cancelled"}}
        )
        if result.modified_count == 1:
            return {"success": True, "message": "Appointment cancelled successfully."}
        return {"success": False, "message": "Appointment not found or already cancelled."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Reschedule an appointment


@app.post("/api/appointments/reschedule/{appointment_id}")
async def reschedule_appointment(appointment_id: str, new_date: str, new_time: str):
    try:
        result = db.appointments.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {"date": new_date, "time": new_time,
                      "status": "Pending Confirmation"}}
        )
        if result.modified_count == 1:
            return {"success": True, "message": "Appointment rescheduled successfully."}
        return {"success": False, "message": "Appointment not found or could not be rescheduled."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Debug endpoint to inspect database state from the app's perspective


@app.get("/api/debug")
async def api_debug():
    try:
        cols = db.list_collection_names()
        tickets_count = db.tickets.count_documents(
            {}) if "tickets" in cols else 0
        appts_count = db.appointments.count_documents(
            {}) if "appointments" in cols else 0

        latest_ticket = None
        if tickets_count > 0:
            doc = db.tickets.find_one({}, sort=[("_id", -1)])
            if doc:
                # convert ObjectId fields to strings for JSON
                doc["_id"] = str(doc["_id"])
                if "attachment_id" in doc:
                    doc["attachment_id"] = str(doc["attachment_id"])
                latest_ticket = doc

        return {
            "db": db.name,
            "collections": cols,
            "tickets_count": tickets_count,
            "appointments_count": appts_count,
            "latest_ticket": latest_ticket,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Return list of tickets (optionally filter by status)
@app.get("/api/tickets")
async def api_tickets(status: str | None = None):
    try:
        query = {"status": {"$ne": "Cancelled"}}  # Exclude cancelled tickets
        if status:
            query["status"] = status
        docs = list(db.tickets.find(query).sort([("_id", -1)]))
        out = []
        for d in docs:
            d["_id"] = str(d["_id"])
            d["date_created"] = d.get("date_created", "Unknown")
            d["last_updated"] = d.get("last_updated", "Unknown")
            d["assigned_staff"] = d.get("assigned_staff", "Not Assigned Yet")
            if "attachment_id" in d:
                d["attachment_id"] = str(d["attachment_id"])
            out.append(d)
        return {"count": len(out), "tickets": out}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Return list of appointments; if upcoming=true, only return date >= today


@app.get("/api/appointments")
async def api_appointments(upcoming: bool = False):
    try:
        # Exclude cancelled appointments
        query = {"status": {"$ne": "Cancelled"}}
        if upcoming:
            today = date.today().isoformat()
            query["date"] = {"$gte": today}
        docs = list(db.appointments.find(
            query).sort([("date", 1), ("time", 1)]))
        out = []
        for d in docs:
            d["_id"] = str(d["_id"])
            d["status"] = d.get("status", "Pending")
            if d["status"] == "Confirmed":
                appointment_date = datetime.strptime(
                    d["date"], "%Y-%m-%d").date()
                days_left = (appointment_date - date.today()).days
                d["countdown"] = f"In {days_left} days" if days_left > 0 else "Today"
            d["location_mode"] = d.get("location_mode", "Unknown")
            if "attachment_id" in d:
                d["attachment_id"] = str(d["attachment_id"])
            out.append(d)
        return {"count": len(out), "appointments": out}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Download GridFS attachment by id
@app.get("/api/attachment/{file_id}")
async def api_attachment(file_id: str):
    try:
        grid_out = fs.get(ObjectId(file_id))
        data = grid_out.read()
        return StreamingResponse(io.BytesIO(data), media_type=(grid_out.content_type or "application/octet-stream"), headers={"Content-Disposition": f"attachment; filename=\"{grid_out.filename or file_id}\""})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)


# Adding an endpoint to fetch user details

@app.get("/api/user")
async def get_user_details(request: Request):
    try:
        user = request.session.get("user")
        print("[DEBUG] User details from session:", user)  # Debug log
        if user:
            return {
                "full_name": user.get("full_name"),
                "email": user.get("email"),
                "role": user.get("role")
            }
        return JSONResponse({"error": "User not logged in"}, status_code=401)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Adding an API endpoint to fetch stats & knowledge base


@app.get("/api/stats")
async def get_stats():
    try:
        knowledge_articles_count = db.knowledge_base.count_documents({})
        departments_count = db.departments.count_documents(
            {"status": "active"})
        total_users_count = db.users.count_documents({})
        upcoming_appointments_count = db.appointments.count_documents({
            "status": {"$ne": "Cancelled"},
            "date": {"$gte": date.today().isoformat()}
        })

        return {
            "knowledge_articles": knowledge_articles_count,
            "departments": departments_count,
            "total_users": total_users_count,
            "upcoming_appointments": upcoming_appointments_count
        }
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {
            "knowledge_articles": 0,
            "departments": 0,
            "total_users": 0,
            "upcoming_appointments": 0
        }


@app.get("/api/knowledge_base")
async def get_knowledge_base():
    try:
        articles = list(db.knowledge_base.find({}, {"_id": 0}))
        return {"articles": articles}
    except Exception as e:
        print(f"Error fetching knowledge base articles: {e}")
        return {"articles": []}


@app.post("/api/knowledge_base")
async def add_knowledge_article(request: Request):
    data = await request.json()
    category = data.get("category")
    title = data.get("title")
    url = data.get("url")

    if not category or not title or not url:
        return JSONResponse({"error": "All fields are required."}, status_code=400)

    try:
        # Extract content from the URL automatically
        from extract_web_content_to_mongo import extract_page, save_to_db
        article = extract_page(url, category, title)
        if not article:
            return JSONResponse({"error": "Failed to fetch content from URL."}, status_code=400)

        # Save to MongoDB
        save_to_db(article)
        return JSONResponse({"message": "Article added successfully."}, status_code=201)
    except Exception as e:
        print(f"Error adding article: {e}")
        return JSONResponse({"error": "Internal server error."}, status_code=500)

import urllib.parse # 👈 Add this import at the top of main.py

# NO new import needed for this part

# Helper function to find and format department info
def get_department_answer(question: str):
    qlow = question.lower()

    # 1) detect if the user even asked for LOCATION-ish stuff
    location_triggers = [
        "where is", "where's", "location", "locate", "map",
        "building", "office", "address", "room", "directions",
        "how do i get", "how to get", "reach", "nearby"
    ]
    has_location_intent = any(trigger in qlow for trigger in location_triggers)

    # if the user did NOT ask for location-ish info, don't hijack the answer
    if not has_location_intent:
        return None

    # 2) now do your old dept lookup
    try:
        all_depts = list(db.departments.find(
            {"status": "active"},
            {"short_name": 1, "name": 1, "building": 1, "office_location": 1}
        ))
        if not all_depts:
            return None
    except Exception as e:
        print(f"[Department Intent] Failed to fetch departments from Mongo: {e}")
        return None

    matched_dept_doc = None
    for dept in all_depts:
        short_name = (dept.get("short_name") or "").lower()
        full_name = (dept.get("name") or "").lower()

        if short_name and short_name in qlow:
            matched_dept_doc = db.departments.find_one({"_id": dept["_id"]})
            break
        if full_name and full_name in qlow:
            matched_dept_doc = db.departments.find_one({"_id": dept["_id"]})
            break

    if not matched_dept_doc:
        return None

    # 3) build the address + map action like you already do
    dept_name = matched_dept_doc.get("name", "the department")
    dept_building = matched_dept_doc.get("building", "")
    destination_address = f"{dept_name}, {dept_building}, Corpus Christi, TX"

    # 4) final payload (same shape you were returning)
    answer_text_lines = [f"The {dept_name} is in **{dept_building}**."]
    if matched_dept_doc.get("office_location"):
        answer_text_lines.append(f"Office/room: **{matched_dept_doc['office_location']}**")

    answer_text_lines.append("Tap the button below to open the map.")
    answer_text = "\n".join(answer_text_lines)

    return {
        "answer": answer_text,
        "source": "department_intent",
        "suggest_live_chat": False,
        "suggested_followups": [
            {
                "label": "Show 3D Walking Map",
                "payload": {
                    "type": "action",
                    "action": "show_map",
                    "destination": destination_address,
                },
            }
        ],
    }

# ======================================================================================
#                                  BOT ENDPOINT
# ======================================================================================
@app.post("/chat_question")
async def chat_question(
    request: Request,
    question: str = Form(...),
    mode: str = Form("general"),
    student_email: str | None = Form(None),
):
    # 👇 figure out who the student is
    student_email = get_student_email_from_request(request, student_email)

    # 👇 if frontend set "student" → use our Mongo filtering
    if mode == "student":
        # This part remains the same
        response_dict = await answer_from_student_scope(request, question, student_email)
        return response_dict

    # ==========================================================
    # 👇 MODIFICATION FOR "general" BOT
    # ==========================================================
    
    # 1. First, try to answer using our new "Department Intent"
    department_answer = get_department_answer(question)
    
    # 1. First, try to answer using our new "Department Intent"
    department_answer_dict = get_department_answer(question) # 👈 RENAMED
    
    if department_answer_dict: # 👈 RENAMED
        print("[General Bot] Answered using Department Intent")
        # We found a precise answer! Return the whole dictionary.
        return department_answer_dict # 👈 NEW

    # 2. If no department was found, fall back to your OLD RAG logic
    print("[General Bot] No department intent. Trying RAG pipeline...")
    from rag_pipeline import get_answer
    answer, _ = get_answer(question)

    chips, suggest_live_chat, fu_source = build_llm_style_followups(
        user_question=question,
        answer_text=answer or "",
        k=4
    )

    if suggest_live_chat:
        chips.append({"label": "Talk to an admin", "payload": {
                     "type": "action", "action": "escalate"}})

    resp = {
        "answer": answer,
        "suggest_live_chat": suggest_live_chat,
        "suggested_followups": chips
    }
    if DEBUG_FOLLOWUPS:
        resp["followup_generator"] = fu_source
    return resp

import re
from bson import ObjectId

def simple_score(text: str, q: str) -> int:
    score = 0
    qwords = re.findall(r"\w+", q.lower())
    tlower = text.lower()
    for w in qwords:
        if w in tlower:
            score += 1
    return score


import json
import re
from fastapi import Request
from bson import ObjectId

async def answer_from_student_scope(request: Request, question: str, student_email: str | None):
    # Helper to create the standard response shape
    def create_response(answer: str, followups: list = None):
        return {
            "answer": answer,
            "suggested_followups": followups if followups is not None else [],
            "suggest_live_chat": False
        }

    if not student_email:
        return create_response("I couldn’t find your student account. Please log in again.")

    qlow = question.lower()

    # --- ✨ Intent A: Show Answers for a Remembered Quiz ---
    is_answer_request = any(k in qlow for k in ["show answer", "what are the answers", "reveal solution", "give answers"])
    if is_answer_request:
        last_quiz = request.session.get("last_quiz")
        if not last_quiz:
            return create_response("I haven't given you a quiz yet. Ask me to 'generate a quiz' first!")
        print("[Student Bot] Showing answers for stored quiz...")
        formatted_answers = ["Here are the answers and explanations for the last quiz:"]
        for i, q in enumerate(last_quiz.get('quiz', []), 1):
            formatted_answers.append(f"\n**{i}. {q.get('question')}**")
            formatted_answers.append(f"   **Answer:** {q.get('answer')}")
            formatted_answers.append(f"   **Explanation:** {q.get('explanation', 'No explanation provided.')}")
        request.session.pop("last_quiz", None)
        return create_response("\n".join(formatted_answers))

    # --- 1. Identify Student's Courses ---
    regs = list(db.registrations.find({"student_email": student_email}))
    if not regs:
        return create_response("You don’t have any registered courses right now.")

    course_ids = [ObjectId(r["course_id"]) for r in regs]
    student_courses = list(db.courses.find({"_id": {"$in": course_ids}}))

    # --- Intent B: List Courses ---
    if "list" in qlow and "course" in qlow:
        lines = ["Here are your registered courses:"]
        for c in student_courses:
            lines.append(f"- {c.get('title')} ({c.get('details')}) • {c.get('term')}")
        return create_response("\n".join(lines))

    # --- 2. Match a Single Course from the Question ---
    matched_course = None
    for c in student_courses:
        title = (c.get("title") or "").lower()
        details = (c.get("details") or "").lower()
        code = details.split(",")[0] if details else ""
        if (title and title in qlow) or (code and code.lower() in qlow):
            matched_course = c
            break

    if not matched_course:
        texts_for_student = list(db.course_materials_text.find({
            "course_id": {"$in": course_ids}
        }))
        course_ids_with_text = {t["course_id"] for t in texts_for_student}
        if len(course_ids_with_text) == 1:
            only_cid = list(course_ids_with_text)[0]
            matched_course = next((c for c in student_courses if c["_id"] == only_cid), None)

    if not matched_course:
        return create_response("You’re in student mode. To ask about course content, please include the course name or code (e.g. 'explain topic from Human-Computer Interaction').")

    # --- 3. Get Course Info & Materials ---
    staff_emails = matched_course.get("staff_emails") or []
    staff_line = f"Instructor(s): {', '.join(staff_emails)}" if staff_emails else "Instructor not listed."
    base_line = (
        f"{matched_course.get('title')} ({matched_course.get('details')}) "
        f"Term: {matched_course.get('term')} "
        f"Schedule type: {matched_course.get('schedule_type')} "
        f"{staff_line}"
    )

    texts = list(db.course_materials_text.find({"course_id": matched_course["_id"]}))
    if texts:  # 👈 fixed from "if mats:"
        for m in texts:  # 👈 fixed from "for m in mats:"
            title = m.get("title", "").lower()
            desc = m.get("description", "").lower()
            if (title and title in qlow) or (desc and desc in qlow):
                link_part = f"Link: {m['file_path']}" if m.get("file_path") else ""
                return create_response(
                    f"{m.get('title','Material')}: {m.get('description','')} {link_part}".strip()
                )

    if not texts:
        mats = list(db.course_materials.find({"course_id": matched_course["_id"], "visible": True}))
        if not mats:
            return create_response(base_line + "\n\nNo materials have been uploaded for this course yet.")
        links = []
        for m in mats:
            if m.get("file_name"):
                links.append(f"- {m['title']}: /uploads/{m['file_name']}")
            elif m.get("external_url"):
                links.append(f"- {m['title']}: {m['external_url']}")
        return create_response(
            base_line
            + "\n\nI found material(s), but they are not text-indexed, so I can't answer questions about them yet:\n"
            + "\n".join(links)
        )

    # --- 4. Find Best Text Context ---
    best_text_doc = None
    best_score = -1
    for t in texts:
        s = simple_score(t.get("text", ""), qlow)
        if s > best_score:
            best_text_doc = t
            best_score = s

    if not best_text_doc:
        return create_response(base_line + "\n\nI found this course, but couldn't match your question to any specific material.")

    context = (best_text_doc.get("text") or "")[:8000]

    # --- 5. Intent-based LLM Calls ---
    is_quiz_request = any(k in qlow for k in ["quiz", "test me", "mcq", "generate questions", "practice problem"])
    is_flashcard_request = any(k in qlow for k in ["flashcard", "make flashcards", "key terms"])
    is_summary_request = any(k in qlow for k in ["summarize", "summary", "tl;dr", "give me the gist"])

    try:
        if is_quiz_request:
            num_questions = 3
            m = re.search(r"(\d+)\s+(quiz|questions)", qlow)
            if m:
                num_questions = int(m.group(1))

            print(f"[Student Bot] Generating {num_questions} quiz questions...")
            system_prompt = f"""
            You are a teaching assistant. Based on the provided course material, generate {num_questions} multiple-choice quiz questions.
            Each question must have 4 options and a brief explanation for the correct answer.
            Return ONLY a valid JSON object in the format: {{"quiz": [...]}}
            """
            user_prompt = f"Course Material:\n{context}"

            raw_json = llm_complete(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=FOLLOWUP_MODEL,
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            quiz_data = json.loads(raw_json)
            request.session["last_quiz"] = quiz_data

            formatted_questions = [f"Okay, I've generated {len(quiz_data.get('quiz', []))} practice questions. Here they are:"]
            for i, q in enumerate(quiz_data.get('quiz', []), 1):
                formatted_questions.append(f"\n**{i}. {q.get('question')}**")
                for opt in q.get("options", []):
                    formatted_questions.append(f"   - {opt}")

            followups = [{"label": "Show me the answers", "payload": {"type": "faq", "query": "show me the answers"}}]
            return create_response("\n".join(formatted_questions), followups)

        elif is_flashcard_request:
            num_cards = 5
            m = re.search(r"(\d+)\s+(flashcard|terms)", qlow)
            if m:
                num_cards = int(m.group(1))

            print(f"[Student Bot] Generating {num_cards} flashcards...")
            system_prompt = f"""
            You are a teaching assistant. Based on the provided material, generate {num_cards} key terms and their definitions as flashcards.
            Return ONLY a valid JSON object in the format: {{"flashcards": [{{"front": "Term", "back": "Definition"}}...]}}
            """
            user_prompt = f"Course Material:\n{context}"

            raw_json = llm_complete(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=FOLLOWUP_MODEL,
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            card_data = json.loads(raw_json)
            formatted_cards = [f"Here are {len(card_data.get('flashcards', []))} flashcards based on the material:"]
            for card in card_data.get('flashcards', []):
                formatted_cards.append(f"\n**{card.get('front')}**")
                formatted_cards.append(f"   - {card.get('back')}")
            return create_response("\n".join(formatted_cards))

        elif is_summary_request:
            print(f"[Student Bot] Generating summary...")
            system_prompt = "You are a teaching assistant. Summarize the provided course material in a few key bullet points."
            user_prompt = f"Course Material:\n{context}"

            answer = llm_complete(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=FOLLOWUP_MODEL,
                temperature=0.2,
                max_tokens=500
            )
            return create_response(answer)

        else:
            print(f"[Student Bot] Answering standard question...")
            system_prompt = """
            You are a helpful and clever course assistant.
            1. Ground your answer strictly in the provided course material.
            2. You can (and should) rephrase, summarize, and explain concepts in a helpful, conversational way.
            3. If the user asks for an example, or if an example would help explain, create a simple, clear example relevant to the topic.
            4. If the question is completely unrelated to the material, state that you can only answer questions about that course's content.
            """
            user_prompt = f"Question: {question}\n\nCourse material:\n{context}"

            answer = llm_complete(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=FOLLOWUP_MODEL,
                temperature=0.4,
                max_tokens=500
            )

            followups = [
                {"label": "Quiz me on this topic", "payload": {"type": "faq", "query": f"generate quiz on {question}"}},
                {"label": "Make flashcards for this", "payload": {"type": "faq", "query": f"make flashcards for {question}"}},
                {"label": "Summarize this topic", "payload": {"type": "faq", "query": f"summarize {question}"}}
            ]
            return create_response(answer, followups)

    except Exception as e:
        print(f"[Student Bot] LLM call failed: {e}")
        return create_response(f"I ran into an error trying to process that: {e}. Please try a different question.")
  
def format_course_answer(c):
    return (
        f"{c.get('title')} ({c.get('details')})\n"
        f"Term: {c.get('term')}\n"
        f"Schedule type: {c.get('schedule_type')}\n"
        f"Hours: {c.get('hours')}"
    )

# Streaming version of chat_question


@app.post("/chat_question_stream")
async def chat_question_stream(question: str = Form(...)):
    from rag_pipeline import get_answer_stream

    async def event_generator():
        # Collect full answer for followup generation
        full_answer = ""

        # Stream the answer chunks
        for chunk in get_answer_stream(question):
            full_answer += chunk
            # Send each chunk as SSE (Server-Sent Events)
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        # Generate followups after answer is complete
        chips, suggest_live_chat, fu_source = build_llm_style_followups(
            user_question=question,
            answer_text=full_answer or "",
            k=4
        )

        if suggest_live_chat:
            chips.append({"label": "Talk to an admin", "payload": {
                         "type": "action", "action": "escalate"}})

        # Send followups
        followup_data = {
            "type": "followups",
            "suggest_live_chat": suggest_live_chat,
            "suggested_followups": chips
        }

        if DEBUG_FOLLOWUPS:
            followup_data["followup_generator"] = fu_source

        yield f"data: {json.dumps(followup_data)}\n\n"

        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ======================================================================================
#                           TICKET CREATION FROM CHATBOT
# ======================================================================================

class TicketAnalysisRequest(BaseModel):
    message: str


class TicketCreateRequest(BaseModel):
    subject: str
    category: str
    priority: str
    description: str
    student_name: str = ""
    student_email: str = ""


@app.post("/api/analyze_ticket")
async def analyze_ticket_request(request: TicketAnalysisRequest, user: dict = Depends(get_current_user)):
    """
    Analyzes user's message using LLM to extract ticket information
    """
    try:
        from rag_pipeline import get_answer
        import re

        # Use LLM to analyze the message
        analysis_prompt = f"""
        Analyze the following user message and extract ticket information.
        
        User message: "{request.message}"
        
        Extract the following:
        1. Subject: A brief subject line (max 100 chars)
        2. Category: One of (Technical Support, Academic, Financial, Housing, Registration, Other)
        3. Priority: One of (Low, Medium, High) - based on urgency in the message
        4. A clear description of the issue
        
        Respond in this exact format:
        SUBJECT: [subject]
        CATEGORY: [category]
        PRIORITY: [priority]
        DESCRIPTION: [description]
        """

        # Get LLM analysis
        answer, _ = get_answer(analysis_prompt)

        # Parse the response
        subject_match = re.search(r'SUBJECT:\s*(.+)', answer)
        category_match = re.search(r'CATEGORY:\s*(.+)', answer)
        priority_match = re.search(r'PRIORITY:\s*(.+)', answer)
        description_match = re.search(
            r'DESCRIPTION:\s*(.+)', answer, re.DOTALL)

        # Extract values or use defaults
        subject = subject_match.group(1).strip(
        ) if subject_match else "Support Request"
        category = category_match.group(
            1).strip() if category_match else "Other"
        priority = priority_match.group(
            1).strip() if priority_match else "Medium"
        description = description_match.group(
            1).strip() if description_match else request.message

        # Validate category
        valid_categories = ["Technical Support", "Academic",
                            "Financial", "Housing", "Registration", "Other"]
        if category not in valid_categories:
            category = "Other"

        # Validate priority
        valid_priorities = ["Low", "Medium", "High"]
        if priority not in valid_priorities:
            priority = "Medium"

        return {
            "subject": subject[:100],  # Limit to 100 chars
            "category": category,
            "priority": priority,
            "description": description
        }
    except Exception as e:
        print(f"Error analyzing ticket: {e}")
        # Fallback to basic extraction
        return {
            "subject": "Support Request",
            "category": "Other",
            "priority": "Medium",
            "description": request.message
        }


@app.post("/api/tickets")
async def create_ticket(ticket: TicketCreateRequest, user: dict = Depends(get_current_user)):
    """
    Creates a new ticket from chatbot
    """
    try:
        # Get student information from session
        student_email = user.get("email", "")
        student_name = user.get("full_name", "")

        # Create ticket document
        ticket_doc = {
            "student_email": student_email,
            "student_name": student_name,
            "subject": ticket.subject,
            "category": ticket.category,
            "priority": ticket.priority,
            "description": ticket.description,
            "status": "Open",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "assigned_staff": None,
            "assigned_to_name": None
        }

        # Insert into database
        result = db.tickets.insert_one(ticket_doc)

        return {
            "success": True,
            "ticket_id": str(result.inserted_id),
            "message": "Ticket created successfully"
        }
    except Exception as e:
        print(f"Error creating ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Google OAuth2 Routes ----------

@app.get("/login/google")
async def login_with_google(request: Request):
    redirect_uri = "http://localhost:8000/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")

    if user_info:
        # Check if user exists in the database
        user = users_collection.find_one({"email": user_info["email"]})

        if not user:
            # Register the user if they don't exist
            users_collection.insert_one({
                "full_name": user_info.get("name"),
                "email": user_info.get("email"),
                "role": "guest",  # Default role for Google SSO users
                "created_at": datetime.utcnow()
            })

        # Store user info in the session
        request.session["user"] = {
            "full_name": user_info.get("name"),
            "email": user_info.get("email"),
            "role": user.get("role", "guest") if user else "guest"
        }

        # Redirect to the appropriate dashboard
        return RedirectResponse(url="/guest_home")

    return RedirectResponse(url="/login")

# Ensure session is completely cleared on logout


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()  # Clear the session completely
    # Debug log to confirm session is empty
    print("[DEBUG] Session after clearing:", request.session)
    return RedirectResponse(url="/login")

# API endpoint to fetch courses by term


@app.get("/api/courses/{term}")
def get_courses(term: str):
    courses = list(db.courses.find({"term": term}))
    return convert_objectid_to_str(courses)

# Helper function to convert ObjectId to string


def convert_objectid_to_str(doc):
    if isinstance(doc, list):
        return [convert_objectid_to_str(d) for d in doc]
    elif isinstance(doc, dict):
        return {k: convert_objectid_to_str(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    return doc

# Define a Pydantic model for course registration


class CourseRegistration(BaseModel):
    student_email: str
    course_id: str
    term: str

# Updated API endpoint to register a student for a course


@app.post("/api/register_course")
def register_course(registration: CourseRegistration):
    try:
        # Validate the registration data
        registration_data = registration.dict()
        db.registrations.insert_one(registration_data)
        return {"message": "Registration successful"}
    except ValidationError as e:
        return JSONResponse({"error": "Invalid registration data", "details": e.errors()}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# API endpoint to fetch registered courses for a student


@app.get("/api/registered_courses/{student_email}")
def get_registered_courses(student_email: str):
    # 1) get all registrations for this student
    registrations = list(db.registrations.find(
        {"student_email": student_email}))

    if not registrations:
        return []

    # 2) collect course_ids from regs
    course_ids = []
    for reg in registrations:
        cid = reg.get("course_id")
        if cid:
            course_ids.append(ObjectId(cid))

    # 3) fetch all those courses in ONE query
    courses = list(db.courses.find({"_id": {"$in": course_ids}}))
    # make a dict for quick lookup
    courses_by_id = {str(c["_id"]): c for c in courses}

    # 4) fetch all materials for these courses
    materials = list(
        db.course_materials.find(
            {"course_id": {"$in": course_ids}, "visible": True}
        )
    )
    # group materials by course_id (as string)
    mats_by_course: dict[str, list] = {}
    for m in materials:
        cid_str = str(m["course_id"])
        m["_id"] = str(m["_id"])
        m["course_id"] = cid_str
        mats_by_course.setdefault(cid_str, []).append(m)

    # 5) build final response
    registered_courses = []
    for reg in registrations:
        reg["_id"] = str(reg["_id"])
        cid_str = reg.get("course_id")

        # attach course details
        course = courses_by_id.get(cid_str)
        if course:
            reg["course_details"] = {
                "title": course.get("title", "N/A"),
                "details": course.get("details", "N/A"),
                "hours": course.get("hours", "N/A"),
                "crn": course.get("crn", "N/A"),
                "schedule_type": course.get("schedule_type", "N/A"),
                "grade_mode": course.get("grade_mode", "N/A"),
                "level": course.get("level", "N/A"),
                "part_of_term": course.get("part_of_term", "N/A"),
            }
        else:
            reg["course_details"] = {}

        # 👇 NEW: attach materials for this course
        reg["materials"] = mats_by_course.get(cid_str, [])

        registered_courses.append(reg)

    return registered_courses

# API endpoint to fetch student data


@app.get("/api/student/{email}")
def get_student(email: str):
    student = db.users.find_one({"email": email})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student["_id"] = str(student["_id"])  # Convert ObjectId to string
    return student

# Define a Pydantic model for student data


class StudentUpdate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str
    marital_status: str
    legal_sex: str
    email: str
    phone_number: str
    address: str


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("update_student")

# Add a test log at the start of the file to verify logging
logger.info("Test log: Logging is working.")

# Updated API endpoint to update student data with full_name auto-generation


@app.put("/api/student/{email}")
def update_student(email: str, student_data: StudentUpdate):
    try:
        logger.info(f"Received update request for email: {email}")
        logger.info(f"Request payload: {student_data.dict()}")

        # Generate full_name separately
        full_name = f"{student_data.first_name} {student_data.last_name}".strip()

        # Update the database with the student data and full_name
        update_data = student_data.dict()
        update_data["full_name"] = full_name

        result = db.users.update_one(
            {"email": email}, {"$set": update_data}, upsert=True)
        if result.modified_count == 0 and not result.upserted_id:
            logger.error("Failed to update student data in the database.")
            raise HTTPException(
                status_code=400, detail="Failed to update student data")

        logger.info("Student data updated successfully.")
        return {"message": "Student data updated successfully"}
    except Exception as e:
        logger.exception("An error occurred while updating student data.")
        raise HTTPException(status_code=500, detail="Internal server error")

# API endpoint to fetch registered classes for a student


@app.get("/api/student/{email}/registered_classes")
def get_registered_classes(email: str):
    registrations = list(db.registrations.find({"student_email": email}))
    registered_classes = []

    for registration in registrations:
        course = db.courses.find_one(
            {"_id": ObjectId(registration["course_id"])})
        if course:
            course["_id"] = str(course["_id"])  # Convert ObjectId to string
            registration["course_details"] = course
        # Convert ObjectId to string
        registration["_id"] = str(registration["_id"])
        registered_classes.append(registration)

    return registered_classes

# API endpoint to get all staff members (for admin to assign tickets)


@app.get("/api/staff")
def get_all_staff():
    try:
        staff_members = list(db.users.find(
            {"role": "staff", "status": "active"},
            {"password": 0}  # Exclude password from response
        ))
        return convert_objectid_to_str(staff_members)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to get staff by department


@app.get("/api/staff/department/{department}")
def get_staff_by_department(department: str):
    try:
        staff_members = list(db.users.find(
            {"role": "staff", "department": department, "status": "active"},
            {"password": 0}
        ))
        return convert_objectid_to_str(staff_members)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to assign ticket to staff


@app.put("/api/tickets/{ticket_id}/assign")
def assign_ticket(ticket_id: str, staff_email: str):
    try:
        from bson import ObjectId

        # Verify staff exists
        staff = db.users.find_one({"email": staff_email, "role": "staff"})
        if not staff:
            raise HTTPException(
                status_code=404, detail="Staff member not found")

        # Update ticket
        result = db.tickets.update_one(
            {"_id": ObjectId(ticket_id)},
            {
                "$set": {
                    "assigned_to": staff_email,
                    "assigned_to_name": staff.get("full_name"),
                    "status": "assigned",
                    "assigned_at": datetime.now().isoformat()
                }
            }
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return {"message": "Ticket assigned successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to update ticket (status and/or assigned staff)


@app.put("/api/tickets/{ticket_id}")
async def update_ticket(ticket_id: str, request: Request):
    try:
        from bson import ObjectId

        data = await request.json()
        status = data.get("status")
        assigned_staff = data.get("assigned_staff")

        update_fields = {
            "last_updated": datetime.now().isoformat()
        }

        if status:
            update_fields["status"] = status

        if assigned_staff and assigned_staff != "":
            # Verify staff exists
            staff = db.users.find_one(
                {"email": assigned_staff, "role": "staff"})
            if staff:
                update_fields["assigned_staff"] = assigned_staff
                update_fields["assigned_to_name"] = staff.get("full_name")
                update_fields["assigned_at"] = datetime.now().isoformat()

        # Update ticket
        result = db.tickets.update_one(
            {"_id": ObjectId(ticket_id)},
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return {"message": "Ticket updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DEPARTMENTS MANAGEMENT ====================

# Get all departments
@app.get("/api/departments")
async def get_departments(status: str | None = None):
    try:
        query = {}
        if status:
            query["status"] = status
        else:
            query["status"] = "active"  # Default to active departments

        departments = list(db.departments.find(query).sort("name", 1))
        return convert_objectid_to_str(departments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get all students for user management


@app.get("/api/students")
async def get_all_students():
    try:
        students = list(db.users.find(
            {"role": "student"},
            {"password": 0}  # Exclude password from response
        ).sort("full_name", 1))
        return convert_objectid_to_str(students)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get single department by ID


@app.get("/api/departments/{department_id}")
async def get_department(department_id: str):
    try:
        department = db.departments.find_one({"department_id": department_id})
        if not department:
            raise HTTPException(status_code=404, detail="Department not found")
        return convert_objectid_to_str(department)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create new department


@app.post("/api/departments")
async def create_department(request: Request):
    try:
        data = await request.json()

        # Check if department_id already exists
        existing = db.departments.find_one(
            {"department_id": data.get("department_id")})
        if existing:
            raise HTTPException(
                status_code=400, detail="Department ID already exists")

        result = db.departments.insert_one(data)
        return {"message": "Department created successfully", "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update department


@app.put("/api/departments/{department_id}")
async def update_department(department_id: str, request: Request):
    try:
        data = await request.json()

        result = db.departments.update_one(
            {"department_id": department_id},
            {"$set": data}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Department not found")

        return {"message": "Department updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete department (soft delete by setting status to inactive)


@app.delete("/api/departments/{department_id}")
async def delete_department(department_id: str):
    try:
        result = db.departments.update_one(
            {"department_id": department_id},
            {"$set": {"status": "inactive"}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Department not found")

        return {"message": "Department deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/materials/all")
async def get_all_materials(request: Request):
    user = request.session.get("user")
    if not user or user.get("role") not in ("staff", "admin"):
        raise HTTPException(403, "Not allowed")

    mats = list(db.course_materials.find({}).sort("uploaded_at", -1))
    # get titles for each course
    course_ids = list({m["course_id"] for m in mats})
    courses = {c["_id"]: c for c in db.courses.find(
        {"_id": {"$in": course_ids}})}

    out = []
    for m in mats:
        m["_id"] = str(m["_id"])
        cid = m["course_id"]
        m["course_id"] = str(cid)
        course = courses.get(cid)
        m["course_title"] = course["title"] if course else "Unknown course"
        out.append(m)
    return out


# --- 1. return ALL courses (used as fallback by staff page) ---

@app.get("/api/courses")
def get_all_courses():
    courses = list(db.courses.find({}))
    return [
        {
            "_id": str(c["_id"]),
            "title": c.get("title"),
            "details": c.get("details"),
            "term": c.get("term"),
        }
        for c in courses
    ]


# --- 2. return ONLY this staff member's courses ---


@app.get("/api/courses/my")
def get_my_courses(request: Request):
    # 1) who is logged in?
    user = request.session.get("user")
    if not user or user.get("role") not in ("staff", "admin"):
        raise HTTPException(status_code=403, detail="Not allowed")

    staff_email = user["email"]

    # 2) DEBUG: show what we are about to search
    print(">>> /api/courses/my called for:", staff_email)
    logger.info("hy fuck you")

    # 3) find courses where this email is in staff_emails
    # (this matches what we saw in /api/debug/courses)
    cursor = db.courses.find({"staff_emails": staff_email})
    courses = list(cursor)

    # 4) DEBUG: how many did we get?
    print(">>> matched courses:", len(courses))

    # 5) if none, return a helpful payload instead of []
    if not courses:
        sample = list(db.courses.find({}).limit(10))
        print(">>> sample from db (first 10):", sample)
        return {
            "message": "no courses matched staff_emails",
            "staff_email_used": staff_email,
            "note": "this means the query ran, but none of the course docs in THIS database have this staff in staff_emails",
            "sample_courses": [
                {
                    "_id": str(c["_id"]),
                    "title": c.get("title"),
                    "staff_emails": c.get("staff_emails"),
                }
                for c in sample
            ],
        }

    # 6) otherwise return clean list
    return [
        {
            "_id": str(c["_id"]),
            "title": c.get("title"),
            "details": c.get("details"),
            "term": c.get("term"),
        }
        for c in courses
    ]



@app.get("/api/materials/mine")
def get_my_materials(request: Request):
    user = request.session.get("user")
    if not user or user.get("role") != "student":
        raise HTTPException(403, "Not allowed")

    email = user["email"]

    # 1) get student registrations
    regs = list(db.registrations.find({"student_email": email}))
    course_ids = [ObjectId(r["course_id"]) for r in regs]

    if not course_ids:
        return []

    # 2) get materials for these courses
    mats = list(db.course_materials.find({
        "course_id": {"$in": course_ids},
        "visible": True
    }).sort("uploaded_at", -1))

    result = []
    for m in mats:
        result.append({
            "_id": str(m["_id"]),
            "course_id": str(m["course_id"]),
            "course_title": m.get("course_title"),
            "title": m.get("title"),
            "description": m.get("description"),
            "file_name": m.get("file_name"),
            "external_url": m.get("external_url"),
            "uploaded_by": m.get("uploaded_by"),
            "uploaded_at": m.get("uploaded_at").isoformat() if m.get("uploaded_at") else None,
        })
    return result




from fastapi import UploadFile, File, HTTPException
from datetime import datetime
from bson import ObjectId

# staff creates material
import re
# staff creates material

@app.post("/api/materials")
async def create_course_material(
    request: Request,
    course_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    file: UploadFile | None = File(None),
    external_url: str = Form(""),
):
    user = request.session.get("user")
    if not user or user.get("role") not in ("staff", "admin"):
        raise HTTPException(403, "Not allowed")

    course = db.courses.find_one({"_id": ObjectId(course_id)})
    if not course:
        raise HTTPException(404, "Course not found")

    saved_filename = None
    if file and file.filename:
        # sanitize filename
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", file.filename)
        abs_path = os.path.join(UPLOAD_DIR, cleaned)
        with open(abs_path, "wb") as f:
            f.write(await file.read())
        saved_filename = cleaned  # <-- store only this
    saved_path = None
    file_type = None

    # if staff uploaded an actual file
    if file:
        contents = await file.read()
        saved_path = f"uploads/{file.filename}"
        with open(saved_path, "wb") as f:
            f.write(contents)
        file_type = file.content_type or "application/octet-stream"
    elif external_url:
        saved_path = external_url
        file_type = "link"

    doc = {
        "course_id": course["_id"],
        "course_title": course.get("title"),
        "title": title,
        "description": description,
        "uploaded_by": user["email"],
        "uploaded_at": datetime.utcnow(),
        "visible": True,
    }

    if saved_filename:
        doc["file_name"] = saved_filename

    if external_url:
        doc["external_url"] = external_url

    result = db.course_materials.insert_one(doc)
    material_id = result.inserted_id
    # try to extract text if it's a PDF
    try:
        if saved_filename and saved_filename.lower().endswith(".pdf"):
            abs_path = os.path.join(UPLOAD_DIR, saved_filename)
            text = extract_pdf_text(abs_path)  # we'll define this below
            if text:
                db.course_materials_text.insert_one({
                    "material_id": material_id,
                    "course_id": doc["course_id"],
                    "course_title": doc.get("course_title"),
                    "file_name": saved_filename,
                    "text": text,
                })
    except Exception as e:
        print("[WARN] could not extract text from material:", e)



@app.get("/api/debug/courses")
def debug_courses(request: Request):
    user = request.session.get("user")
    # get all courses the backend is ACTUALLY seeing
    courses = list(db.courses.find({}))
    # keep it light
    preview = []
    for c in courses[:10]:
        preview.append({
            "_id": str(c["_id"]),
            "title": c.get("title"),
            "details": c.get("details"),
            "term": c.get("term"),
            "staff_emails": c.get("staff_emails"),
        })
    return {
        "session_user": user,
        "courses_count": len(courses),
        "courses_preview": preview,
    }


# get materials for a course
@app.get("/api/materials/by_course/{course_id}")
async def get_materials_by_course(course_id: str):
    mats = list(db.course_materials.find({
        "course_id": ObjectId(course_id),
        "visible": True
    }).sort("uploaded_at", -1))
    # convert ObjectId -> str
    out = []
    for m in mats:
        m["_id"] = str(m["_id"])
        m["course_id"] = str(m["course_id"])
        out.append(m)
    return out
