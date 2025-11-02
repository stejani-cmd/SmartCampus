# extract_web_content_mongo.py
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import os

# ------------------ MongoDB setup ------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017/smartassist")
client = MongoClient(MONGO_URI)
db = client.smartassist
kb_collection = db.knowledge_base  # collection for articles

# Create unique index on URL to prevent duplicates
kb_collection.create_index("url", unique=True)

# ------------------ Utility functions ------------------
def clean_text(text):
    """Remove extra whitespace and line breaks."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_page(url, category, title):
    """Fetch page content and return a dictionary of article info."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content
    main_content = soup.find("main") or soup.find("div", {"id": "content"}) or soup

    # Extract and clean text
    text = clean_text(main_content.get_text(separator=" ", strip=True))

    data = {
        "category": category,
        "title": title,
        "url": url,
        "content": text
    }
    return data

def save_to_db(article):
    """Insert article into MongoDB if URL does not exist yet."""
    try:
        kb_collection.insert_one(article)
        print(f"✅ Saved: {article['title']}")
    except Exception as e:
        print(f"⚠️ Skipped (probably duplicate): {article['title']} | {e}")

# ------------------ Example usage ------------------
if __name__ == "__main__":
    pages = [
        {
            "url": "https://www.tamucc.edu/admissions/first-time-freshmen.php",
            "category": "Admissions",
            "title": "First-Time Freshmen Requirements"
        },
        {
            "url": "https://www.tamucc.edu/admissions/apply.php",
            "category": "Apply",
            "title": "How to Apply"
        },
        {
            "url": "https://www.tamucc.edu/admissions/admissions-counselors.php",
            "category": "Admissions Counselors",
            "title": "Meet Your Admissions Counselor"
        },
        {
            "url": "https://www.tamucc.edu/admissions/returning-students.php",
            "category": "Returning Students",
            "title": "Returning Student Admissions"
        },
        {
            "url": "https://www.tamucc.edu/admissions/international-freshmen.php",
            "category": "International Freshmen Admissions",
            "title": "International Freshmen Admissions"
        },
        {
            "url": "https://www.tamucc.edu/admissions/international-transfer.php",
            "category": "International Transfer Admissions",
            "title": "International Transfer Admissions"
        },
        {
            "url": "https://www.tamucc.edu/admissions/graduate-students/application-process.php",
            "category": "Graduate Students",
            "title": "Graduate Student Application Process"
        },
        {
            "url": "https://www.tamucc.edu/cost-and-aid/scholarships/index.php",
            "category": "Scholarship & Financial Aid",
            "title": "Enrollment Management Scholarships"
        },
        {
            "url": "https://www.tamucc.edu/academics/colleges/index.php",
            "category": "Colleges",
            "title": "Colleges"
        },
        {
            "url": "https://www.tamucc.edu/business/index.php",
            "category": "Colleges",
            "title": "College of business"
        },
        {
            "url": "https://www.tamucc.edu/education/index.php",
            "category": "Colleges",
            "title": "College of Education and Human Developement"
        },
          {
            "url": "https://www.tamucc.edu/engineering/index.php",
            "category": "Colleges",
            "title": "College of Engineering and Computer Science"
        },
          {
            "url": "https://www.tamucc.edu/science/index.php",
            "category": "Colleges",
            "title": "College of Science"
        },
        {
            "url": "https://www.tamucc.edu/liberal-arts/index.php",
            "category": "Colleges",
            "title": "College of Liberal and Arts"
        },
        {
            "url": "https://www.tamucc.edu/nursing-and-health-sciences/index.php",
            "category": "Colleges",
            "title": "College of Nursing & Health Sciences"
        },
        {
            "url": "https://www.tamucc.edu/samc/index.php",
            "category": "Colleges",
            "title": "School of Arts, Media, & Communication"
        },
        {
            "url": "https://www.tamucc.edu/business/about/index.php",
            "category": "College of Business",
            "title": "About the college of business"
        },
        {
            "url": "https://www.tamucc.edu/business/strategic-plan/index.php",
            "category": "College of Business",
            "title": "College Of Business Strategic Plan"
        },
        {
            "url": "https://www.tamucc.edu/business/students/index.php",
            "category": "College of Business",
            "title": "Student Engagement Plan"
        },
        {
            "url": "https://www.tamucc.edu/business/advising/index.php",
            "category": "College of Business",
            "title": "Advising"
        },
        {
            "url": "https://www.tamucc.edu/business/graduate-programs/index.php",
            "category": "College of Business",
            "title": "Graduate Programs"
        },
        {
            "url": "https://www.tamucc.edu/business/undergraduate-programs/index.php",
            "category": "College of Business",
            "title": "Undergraduate Programs"
        },
        {
            "url": "https://www.tamucc.edu/business/centers-research/index.php",
            "category": "College of Business",
            "title": "Student and Research"
        },
        {
            "url": "https://www.tamucc.edu/business/student-resource-directory/index.php",
            "category": "College of Business",
            "title": "Student Resource Directory"
        },
        {
            "url": "https://www.tamucc.edu/business/student-resource-directory/prospective-student-resources.php",
            "category": "College of Business",
            "title": "Prospective Student Resources"
        },
        {
            "url": "https://www.tamucc.edu/business/news-and-events/index.php",
            "category": "College of Business",
            "title": "News and Events"
        },


    ]

    for page in pages:
        article = extract_page(page["url"], page["category"], page["title"])
        if article:
            save_to_db(article)
