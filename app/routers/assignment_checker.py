# app/routers/assignment_checker.py
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.services.assignment_checker_agents import (
    IngestionAgent,
    RequirementAgent,
    ScoringAgent,
    FeedbackAgent,
)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ingestion_agent = IngestionAgent(upload_dir=UPLOAD_DIR)
requirement_agent = RequirementAgent()
scoring_agent = ScoringAgent()
feedback_agent = FeedbackAgent()


@router.get("/assignment-checker", response_class=HTMLResponse)
async def assignment_checker_page(request: Request):
    return templates.TemplateResponse(
        "assignment_checker.html",
        {"request": request},
    )


@router.post("/api/assignment-checker/score")
async def assignment_checker_api(
    request: Request,
    requirements: str = Form(...),
    student_text: str = Form(""),
    file: UploadFile | None = File(None),
):
    filename = None
    ext = None
    saved_path = None

    if file:
        filename = file.filename
        ext = Path(filename).suffix.lower()
        saved_path = UPLOAD_DIR / filename
        saved_path.write_bytes(await file.read())

    # ingest
    ingest_result = ingestion_agent.run(
        file_path=saved_path,
        ext=ext,
        text_override=student_text if student_text.strip() else None,
    )
    doc_text = ingest_result["text"]
    doc_meta = ingest_result["meta"]

    # requirements
    structured = requirement_agent.run(raw_requirements=requirements)

    # score
    scoring_result = scoring_agent.run(
        doc_text=doc_text,
        doc_meta=doc_meta,
        requirements=structured,
    )

    feedback = feedback_agent.run(
        details=scoring_result["details"],
        notes=doc_meta.get("notes", []),
    )

    # simple extras
    score = scoring_result["score"]
    plagiarism = max(1, 12 - score // 10)
    formatting = min(90, 100 - score + 5)

    return JSONResponse({
        "filename": filename,
        "score": score,
        "plagiarism": plagiarism,
        "formatting": formatting,
        "requirements_count": len(structured),
        "extracted_text_preview": (doc_text or "")[:4000],
        "details": scoring_result["details"],
        "feedback": feedback,
        # 👇 this is the important part now
        "dominant_font_size": doc_meta.get("dominant_font_size"),
        "dominant_font_name": doc_meta.get("dominant_font_name"),
        "font_sizes_counter": doc_meta.get("font_sizes_counter", {}),
        "font_names_counter": doc_meta.get("font_names_counter", {}),
        "notes": doc_meta.get("notes", []),
    })
