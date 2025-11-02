import json
import pymongo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017/smartassist")
client = pymongo.MongoClient(MONGO_URI)
db = client["smartassist"]
courses_collection = db["courses"]

# Function to load courses from JSON file and insert into MongoDB
def load_courses_from_json(file_path: str):
    with open(file_path, 'r') as file:
        courses = json.load(file)
        courses_collection.insert_many(courses)
        print("Courses loaded successfully into MongoDB.")

if __name__ == "__main__":
    load_courses_from_json('courses.json')