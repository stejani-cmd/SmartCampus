# app/db/mongo.py
from pymongo import MongoClient
from app.core.config import settings

client = MongoClient(settings.MONGODB_URI)
db = client[settings.MONGODB_DB]

# user collections
users_collection = db.users

# forum collections
forum_categories = db.forum_categories
forum_posts = db.forum_posts
forum_comments = db.forum_comments
