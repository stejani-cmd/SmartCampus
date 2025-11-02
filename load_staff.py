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

# Function to load staff from JSON file and insert into database
def load_staff_from_json(file_path: str):
    with open(file_path, 'r') as file:
        staff_members = json.load(file)
        
        # Clear existing staff data (optional)
        # db.users.delete_many({"role": "staff"})
        
        for staff in staff_members:
            # Check if staff already exists
            existing_staff = db.users.find_one({"email": staff['email']})
            if not existing_staff:
                db.users.insert_one(staff)
                print(f"Added staff: {staff['full_name']}")
            else:
                print(f"Staff already exists: {staff['full_name']}")
        
        print(f"\nTotal staff members in database: {db.users.count_documents({'role': 'staff'})}")

if __name__ == "__main__":
    load_staff_from_json('staff_data.json')
    print("Staff data loaded successfully!")