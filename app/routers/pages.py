# app/routers/pages.py
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.core.config import settings
from fastapi import APIRouter, Request, Depends
from app.utils import role_required


templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/student_home", response_class=HTMLResponse)
async def student_home(
    request: Request,
    user: dict = Depends(role_required("student")),
):
    return templates.TemplateResponse(
        "student_home.html",
        {"request": request, "user": user},
    )



@router.get("/staff_home", response_class=HTMLResponse)
async def staff_home(
    request: Request,
    user: dict = Depends(role_required("staff")),
):
    return templates.TemplateResponse("staff_home.html", {"request": request, "user": user})


@router.get("/admin_home", response_class=HTMLResponse)
async def admin_home(
    request: Request,
    user: dict = Depends(role_required("admin")),
):
    return templates.TemplateResponse("admin_home.html", {"request": request, "user": user})
