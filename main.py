# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from starlette.middleware.sessions import SessionMiddleware
import os
from app.routers import auth, pages

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
