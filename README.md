# SmartAssist – Campus Services Assistant

SmartAssist is an AI-powered campus services assistant designed to enhance student support and streamline campus interactions at **Texas A&M University–Corpus Christi (TAMU-CC)**.  
It provides students with a centralized, intelligent platform to get answers to common questions, submit service requests, schedule appointments, and receive real-time location guidance — all from a single responsive web interface.

---

## 📚 Table of Contents
1. [Features](#-features)
2. [Goals](#-goals)
3. [Tech Stack](#-tech-stack)
4. [System Requirements](#-system-requirements)
5. [Installation Steps](#-installation-steps)
6. [Testing](#-testing)
7. [Future Updates](#-future-updates)
8. [License](#-license)

---

## 🚀 Features
- **Intelligent Knowledge Base & FAQs** – Searchable database of articles and guides available 24/7.  
- **AI-Powered Chatbot** – Natural-language query handling with escalation to live chat.  
- **Service Request & Ticket Management** – Submit, track, and manage service requests.  
- **Appointment Scheduling** – Book and manage advising or service appointments.  
- **Responsive Web Portal** – Consistent user experience across devices.  
- **Feedback & Surveys** – Built-in rating tools for students to share feedback.  
- **Analytics & Reporting** – Dashboards for staff to monitor ticket resolution and usage.  
- **Location Guidance** – Real-time directions for professor offices, classrooms, and buildings.  
- **Optional Community Forum** – Peer-to-peer Q&A space.  
- **Admin Portal** – Manage professor info, building data, and event venues.  
- **LLM + RAG Integration** – Retrieves professor and building info from the campus database.

---

## 🎯 Goals
1. Resolve **60%+ of student queries** through self-service.  
2. Reduce **average ticket resolution time by 30%**.  
3. Deliver a **mobile-friendly interface** with accessibility compliance.  
4. Provide **analytics dashboards** for service improvement.  
5. Automate location-based questions to **reduce front-desk dependency**.

---

## 🛠️ Tech Stack
- **Frontend:** HTML/CSS/Tailwind (Responsive Web Portal, UI Components)  
- **Backend:** FastAPI (API Development), Python (AI/ML Pipeline)  
- **Database:** MongoDB  
- **AI & NLP:** LLM + Retrieval-Augmented Generation (RAG)
- **Deployment:** Docker + AWS  
- **Version Control:** GitHub / GitLab  

---

## 💻 System Requirements

### Hardware
- Minimum **8 GB RAM** (16 GB recommended)  
- **50 GB** available storage  
- Multi-core processor (**Quad-core or higher**)  
- Stable internet connection  

### Software
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu recommended)  
- **Development Tools:**  
  - Python 3.10 or higher  
  - Git  
  - VS Code / PyCharm / any IDE  
- **Database:** MongoDB (local or cloud instance)  
- **Python Dependencies:**  
  - FastAPI  
  - Jinja2  
  - Pymongo  
  - OpenAI API  
  - Uvicorn  
  - Websocket  
  - Scikit-learn  
  - Torch  
  - SentenceTransformers  

---

## ⚙️ 3. Installation Steps

### 3.1 Clone the Repository
1. Open terminal/command prompt  
2. Clone the SmartAssist repository:  
   ◦ `git clone https://github.com/smit-tejani/SmartAssist-Campus-Services-Assistant.git`  
3. Navigate to the project folder:  
   ◦ `cd SmartAssist-Campus-Services-Assistant`  


---

### 3.2 Backend Setup (AI Models & API)
1. **Create a Python virtual environment:**

   **macOS/Linux:**  
   ◦ `python3 -m venv sa_env`  
   ◦ `source sa_env/bin/activate`  

   **Windows:**  
   ◦ `python3 -m venv sa_env`  
   ◦ `sa_env\Scripts\activate`  

2. **Install Python dependencies:**  
   ◦ `pip install -r requirements.txt`  

3. **Run the FastAPI server:**  
   ◦ `uvicorn main:app --reload`  

4. **Open a browser and navigate to:**  
   ◦ `http://localhost:8000`  

5. **Verify that the Knowledge Base & AI Chatbot are operational.**

---

## 🧪 Testing
The system is tested at multiple levels:
- **Unit Tests** – Chatbot responses, API endpoints.  
- **Integration Tests** – Frontend-backend communication.  
- **System Tests** – End-to-end user scenarios.  
- **User Testing** – Feedback from students and faculty.  

---

UI for ticketing and appointments has been designed and will be integrated in future builds.


**Note:** For the latest deployment instructions, always refer to the `Software Installation Manual` and this README file.
