# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "smartassist"
    TEMPLATES_DIR: str = "templates"
    STATIC_DIR: str = "static"

    # mongo
    MONGODB_URI: str = "mongodb+srv://Manny0715:Manmeet12345@cluster0.1pf6oxg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    MONGODB_DB: str = "smartassist"

settings = Settings()
