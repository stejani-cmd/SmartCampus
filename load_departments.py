import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017/smartassist")
client = MongoClient(MONGO_URI)
db = client["smartassist"]

# Function to load departments from JSON file and insert into database
def load_departments_from_json(file_path: str):
    with open(file_path, 'r') as file:
        departments = json.load(file)
        
        # Clear existing departments data (optional - comment out if you want to keep existing data)
        # db.departments.delete_many({})
        
        for dept in departments:
            # Check if department already exists
            existing_dept = db.departments.find_one({"department_id": dept['department_id']})
            if not existing_dept:
                db.departments.insert_one(dept)
                print(f"Added department: {dept['name']}")
            else:
                print(f"Department already exists: {dept['name']}")
        
        print(f"\nTotal departments in database: {db.departments.count_documents({})}")

if __name__ == "__main__":
    load_departments_from_json('departments_data.json')
    print("Departments data loaded successfully!")
