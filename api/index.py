# api/index.py
import os
import sys

# make repo root importable when this file lives in /api
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# main.py already has: app = FastAPI() and mounts /static + templates
from main import app  # noqa: F401
