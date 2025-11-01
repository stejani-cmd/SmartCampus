# app/routers/auth.py
from fastapi import APIRouter, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.core.config import settings
from app.db.mongo import users_collection
from app.utils import redirect_if_logged_in

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

@router.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    maybe_redirect = redirect_if_logged_in(request)
    if maybe_redirect:
        return maybe_redirect

    return templates.TemplateResponse("login.html", {"request": request})



@router.post("/login")
async def post_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
):
    user = users_collection.find_one({"email": email})

    if user and user["password"] == password and user["role"] == role:
        # put user in session
        request.session["user"] = {
            "full_name": user["full_name"],
            "email": user["email"],
            "role": user["role"],
        }

        # redirect by role (you had this)
        if role == "student":
            return RedirectResponse("/student_home", status_code=302)
        elif role == "staff":
            return RedirectResponse("/staff_home", status_code=302)
        elif role == "admin":
            return RedirectResponse("/admin_home", status_code=302)

    # failed login
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials or role!", "email": email},
    )


@router.get("/register", response_class=HTMLResponse)
async def get_register(request: Request):
    maybe_redirect = redirect_if_logged_in(request)
    if maybe_redirect:
      return maybe_redirect

    return templates.TemplateResponse("register.html", {"request": request})


@router.post("/register", response_class=HTMLResponse)
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
            {
                "request": request,
                "error": "Passwords do not match!",
                "full_name": full_name,
                "email": email,
            },
        )

    if users_collection.find_one({"email": email}):
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Email already registered!",
                "full_name": full_name,
            },
        )

    users_collection.insert_one(
        {
            "full_name": full_name,
            "email": email,
            "password": password,  # TODO: hash
            "role": "student",  # default role
        }
    )

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "message": "Registration successful! Please login."},
    )


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)