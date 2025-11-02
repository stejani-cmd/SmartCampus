from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from bson import ObjectId

from app.core.config import settings
from app.db.mongo import forum_posts, forum_comments, forum_categories
from app.utils import get_current_user

router = APIRouter(prefix="/forum", tags=["forum"])
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)


def oid(x: str):
    return ObjectId(str(x))


@router.get("/", response_class=HTMLResponse)
async def forum_home(request: Request, category: str | None = None):
    q = {}
    if category:
        q["category_slug"] = category

    posts = list(forum_posts.find(q).sort("updated_at", -1).limit(40))
    cats = list(forum_categories.find())
    return templates.TemplateResponse(
        "forum_list.html",
        {
            "request": request,
            "posts": posts,
            "categories": cats,
            "active_category": category,
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def new_post_page(request: Request, user=Depends(get_current_user)):
    cats = list(forum_categories.find())
    return templates.TemplateResponse(
        "forum_new.html",
        {"request": request, "categories": cats, "user": user},
    )


@router.post("/new")
async def create_post(
    request: Request,
    title: str = Form(...),
    body: str = Form(""),
    category_slug: str = Form("general"),
    tags: str = Form(""),
    anonymous: str | None = Form(None),
    user=Depends(get_current_user),
):
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    is_anon = anonymous == "1"
    display_name = "Anonymous" if is_anon else (
        user.get("full_name") or "Student")

    doc = {
        "title": title.strip(),
        "body": body.strip(),
        "category_slug": category_slug,
        "tags": tag_list,
        "author_id": user.get("id") or user.get("_id"),
        "author_name": display_name,
        "anonymous": is_anon,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "views": 0,
    }
    res = forum_posts.insert_one(doc)
    return RedirectResponse(url=f"/forum/{res.inserted_id}", status_code=302)


@router.get("/{post_id}", response_class=HTMLResponse)
async def read_post(post_id: str, request: Request):
    post = forum_posts.find_one({"_id": oid(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    # count view
    forum_posts.update_one({"_id": oid(post_id)}, {"$inc": {"views": 1}})

    comments = list(
        forum_comments.find({"post_id": post_id}).sort("created_at", 1)
    )
    cats = list(forum_categories.find())
    return templates.TemplateResponse(
        "forum_thread.html",
        {
            "request": request,
            "post": post,
            "comments": comments,
            "categories": cats,
        },
    )


@router.post("/{post_id}/comment")
async def add_comment(
    post_id: str,
    body: str = Form(...),
    user=Depends(get_current_user),
):
    post = forum_posts.find_one({"_id": oid(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    body = body.strip()
    if not body:
        return RedirectResponse(url=f"/forum/{post_id}", status_code=302)

    forum_comments.insert_one(
        {
            "post_id": post_id,
            "body": body,
            "author_id": user.get("id") or user.get("_id"),
            "author_name": user.get("full_name") or "Student",
            "created_at": datetime.utcnow(),
        }
    )
    # bump thread
    forum_posts.update_one(
        {"_id": oid(post_id)}, {"$set": {"updated_at": datetime.utcnow()}}
    )
    return RedirectResponse(url=f"/forum/{post_id}", status_code=302)
