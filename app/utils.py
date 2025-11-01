# app/utils.py
from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import RedirectResponse


def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        request.session.clear()
        # redirect to login
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
            detail="Redirecting to login",
        )
    return user


def role_required(required_role: str):
    def _dep(user = Depends(get_current_user)):   # ✅ use Depends here
        if user.get("role") != required_role:
            # you can also send 403 here instead of redirect
            raise HTTPException(
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                headers={"Location": "/login"},
                detail="Forbidden",
            )
        return user
    return _dep



def redirect_if_logged_in(request: Request):
    user = request.session.get("user")
    if not user:
        return None

    role = user.get("role")
    if role == "student":
        return RedirectResponse("/student_home", status_code=302)
    if role == "staff":
        return RedirectResponse("/staff_home", status_code=302)
    if role == "admin":
        return RedirectResponse("/admin_home", status_code=302)

    # fallback
    return RedirectResponse("/", status_code=302)