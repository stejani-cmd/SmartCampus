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
import re
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

from app.routers import auth, pages
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("update_student")

# ---------- env ----------
load_dotenv()
print("[BOOT] USE_LLM_FOLLOWUPS=", os.getenv("USE_LLM_FOLLOWUPS", "1"),
      "MODEL=", os.getenv("FOLLOWUP_MODEL", "gpt-4o-mini"),
      "OPENAI_KEY_PRESENT=", bool(os.getenv("OPENAI_API_KEY")))


app = FastAPI(title=settings.PROJECT_NAME)

# static
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "dev-secret-change-me"),
    session_cookie="smartcampus_session",
)

# routers
app.include_router(pages.router)
app.include_router(auth.router)

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
MONGO_URI = os.getenv(
    "MONGODB_URI", "mongodb+srv://Manny0715:Manmeet12345@cluster0.1pf6oxg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client = MongoClient(MONGO_URI)
db = client.SmartCampus
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


def llm_complete(messages, model="gpt-4o-mini", temperature=0.4, max_tokens=180) -> str:
    """
    Returns assistant text using whichever OpenAI SDK is installed.
    Uses a legacy-safe model when v0 SDK is detected.
    """
    # Try v1 SDK
    try:
        from openai import OpenAI  # v1+
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # if your model supports it, uncomment next line to force JSON
            # response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content.strip()
    except Exception as v1_err:
        # Fallback to legacy v0 SDK
        import openai
        if not getattr(openai, "api_key", None):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        legacy_model = os.getenv("FOLLOWUP_MODEL_LEGACY", "gpt-3.5-turbo")
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

# API endpoint to fetch courses by term


@app.get("/api/courses/{term}")
def get_courses(term: str):
    courses = list(db.courses.find({"term": term}))
    return convert_objectid_to_str(courses)


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

# API endpoint to fetch registered courses for a student


@app.get("/api/registered_courses/{student_email}")
def get_registered_courses(student_email: str):
    registrations = list(db.registrations.find(
        {"student_email": student_email}))
    registered_courses = []

    for registration in registrations:
        course = db.courses.find_one(
            {"_id": ObjectId(registration["course_id"])})
        if course:
            course["_id"] = str(course["_id"])  # Convert ObjectId to string
            registration["course_details"] = {
                "title": course.get("title", "N/A"),
                "details": course.get("details", "N/A"),
                "hours": course.get("hours", "N/A"),
                "crn": course.get("crn", "N/A"),
                "schedule_type": course.get("schedule_type", "N/A"),
                "grade_mode": course.get("grade_mode", "N/A"),
                "level": course.get("level", "N/A"),
                "part_of_term": course.get("part_of_term", "N/A"),
            }
        # Convert ObjectId to string
        registration["_id"] = str(registration["_id"])
        registered_courses.append(registration)

    return registered_courses

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


# ======================================================================================
#                                  BOT ENDPOINT
# ======================================================================================
@app.post("/chat_question")
async def chat_question(question: str = Form(...)):
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
        # "openai" | "fallback" | "fallback_error"
        resp["followup_generator"] = fu_source
    return resp


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
