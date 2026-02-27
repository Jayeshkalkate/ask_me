---

# 🤖 ASK_ME – Smart Document & Knowledge Assistant

🔗 **Live Demo:** [https://ask-me-smart-document-assistant.onrender.com](https://ask-me-smart-document-assistant.onrender.com)
💻 **GitHub Repository:** [https://github.com/Jayeshkalkate/ask_me](https://github.com/Jayeshkalkate/ask_me)

---

## 📌 Project Overview

ASK_ME is an AI-powered Smart Document Assistant that allows users to upload documents, extract text using OCR, and interact with their content through intelligent contextual responses.

The platform is designed to simplify document understanding by enabling users to ask questions about uploaded files and receive structured, AI-driven insights.

This project demonstrates full-stack development, AI integration, OCR implementation, authentication systems, cloud deployment, and scalable backend architecture.

---

## 🎯 Problem Statement

Users often struggle to extract meaningful insights from lengthy documents. Manual reading is time-consuming and inefficient.

ASK_ME solves this problem by:

* Extracting text from documents using OCR
* Structuring document data
* Enabling intelligent Q&A on uploaded content
* Providing a user-friendly interface for document interaction

---

## 🚀 Key Features

### 🔐 User Authentication System

* Secure login & registration
* Session-based authentication
* User-specific document management

### 📄 Document Upload & Management

* Upload PDF/Image documents
* Store and manage multiple files
* Organized document dashboard

### 🔍 OCR Text Extraction

* Integrated Tesseract OCR for text extraction
* Automatic text processing from images & scanned PDFs
* Structured text handling

### 🤖 AI-Based Contextual Q&A

* AI-powered response generation
* Context-aware document interaction
* Intelligent information retrieval

### 📊 Admin & Backend Management

* Admin panel for managing users and documents
* Database handling and monitoring

### ☁️ Cloud Deployment

* Hosted on Render
* Production-ready configuration
* Environment variable management

---

## 🏗️ System Architecture

User → Django Backend → OCR Processing → AI Processing → Database Storage → Response Rendering

The architecture follows a modular backend design ensuring:

* Separation of concerns
* Scalable structure
* Clean API handling
* Optimized performance

---

## 🛠️ Tech Stack

### 🔹 Backend

* Python
* Django
* Django REST Architecture
* PostgreSQL (Production Ready)

### 🔹 Frontend

* HTML5
* CSS3
* Tailwind CSS
* JavaScript

### 🔹 AI & Processing

* Tesseract OCR
* Regex-based Data Extraction
* AI API Integration (Configurable)

### 🔹 DevOps & Deployment

* Render Cloud Deployment
* Git & GitHub
* Environment Variables Management
* Production Settings Configuration

---

## 📂 Project Structure

```
ask_me/
│
├── account/              # User authentication app
├── documents/            # Document handling & OCR logic
├── templates/            # Frontend UI templates
├── static/               # CSS, JS, media files
├── settings.py           # Django configuration
├── requirements.txt      # Project dependencies
└── manage.py             # Django project entry
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Jayeshkalkate/ask_me.git
cd ask_me
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file and configure:

```
SECRET_KEY=your_secret_key
DEBUG=True
DATABASE_URL=your_database_url
```

### 5️⃣ Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6️⃣ Run Server

```bash
python manage.py runserver
```

---

## 📈 Future Enhancements

* Vector database integration for semantic search
* Full AI chatbot mode
* Document summarization
* Multi-language OCR support
* Docker containerization
* CI/CD pipeline automation

---

## 🧠 Learning Outcomes

Through this project, I gained hands-on experience in:

* Full Stack Web Development
* OCR Integration
* Regex-based data extraction
* Secure authentication implementation
* Cloud deployment
* Backend optimization
* AI system integration
* Production-level project structuring

---

## 👨‍💻 Author

**Jayesh Rajendra Kalkate**
B.Tech Computer Engineering (2022–2026)

📧 Email: [kalkatejayesh@gmail.com](mailto:kalkatejayesh@gmail.com)
🔗 LinkedIn: [https://www.linkedin.com/in/jayesh-kalkate-31a250242](https://www.linkedin.com/in/jayesh-kalkate-31a250242)
💻 GitHub: [https://github.com/Jayeshkalkate](https://github.com/Jayeshkalkate)

---

## 📜 License

This project is developed for educational and portfolio purposes.

---

# 🔥 Why This Project Stands Out

ASK_ME is not just a CRUD application — it combines:

* AI + OCR
* Backend architecture
* Secure authentication
* Cloud deployment
* Real-world document intelligence

It represents a practical implementation of modern full-stack and AI-integrated application development.

---
