# seed_forum_categories.py
from pymongo import MongoClient

MONGODB_URI: str = "mongodb+srv://Manny0715:Manmeet12345@cluster0.1pf6oxg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB: str = "smartassist"

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

forum_categories = db.forum_categories

seed = [
    {
        "slug": "general",
        "name": "General",
        "description": "Anything campus-related, announcements, doubts.",
    },
    {
        "slug": "it-support",
        "name": "IT / LMS support",
        "description": "Wi-Fi, LMS, portal, login, email issues.",
    },
    {
        "slug": "hostel",
        "name": "Hostel / Mess",
        "description": "Hostel timings, mess menu, facilities requests.",
    },
    {
        "slug": "events",
        "name": "Events & Clubs",
        "description": "Workshops, hackathons, student club activities.",
    },
    {
        "slug": "placements",
        "name": "Placements & Internships",
        "description": "Drives, resume review, off-campus opportunities.",
    },
]

for cat in seed:
    forum_categories.update_one(
        {"slug": cat["slug"]},
        {"$setOnInsert": cat},
        upsert=True,
    )

print("✅ Forum categories seeded (upserted).")
