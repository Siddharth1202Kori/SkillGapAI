# SkillGap AI 🎯
*Bridging the gap between your skills and real-time job market demands using the power of Retrieval-Augmented Generation (RAG).*

![Screenshot](/frontend/styles.css) *(Add a screenshot here later!)*

SkillGap AI is an intelligent career assistant powered by **Mistral LLM** and **ChromaDB**. It dynamically scrapes live job postings, analyzes them semantically, and compares the market demands against your personal background to generate an actionable, highly tailored career roadmap.

## 🚀 Key Features
* **Live Job Tracking:** Automatically fetches 100% real remote job descriptions from the Remotive API (bypassing HTML scrapers).
* **Data Lake Storage:** Backs up raw un-chunked JSON job data natively to a Secure Supabase Storage Bucket.
* **Semantic Analysis (RAG):** Uses `mistral-embed` to inject job chunks into a localized ChromaDB vector store, enabling high-fidelity cosine similarity context retrieval.
* **Premium Dashboard UI:** An elegant, Notch/Linear-inspired frontend built natively to dynamically parse complex markdown LLM outputs into cleanly rendered interactive insights.

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **LLM & Embeddings:** Mistral AI (`mistral-small-latest` & `mistral-embed`)
* **Vector Database:** ChromaDB 
* **Frameworks:** LangChain, Flask (API backend)
* **Frontend:** Vanilla JS/CSS, HTML5, marked.js
* **Storage:** Supabase

---

## 💻 Local Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/skillgap-ai.git
cd skillgap-ai
```

### 2. Set up the Python virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)
Create a `.env` file in the root directory and add securely required API Keys:
```env
# Mistral API (For embeddings and LLM generation)
MISTRAL_API_KEY=your_mistral_api_key_here

# Supabase Storage Bucket
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here

# Local Vector store config
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=indeed_jobs
```

---

## 🏃‍♂️ Running the Application

SkillGap AI operates using a **Flask Wrapper API** connected seamlessly to a pristine HTML frontend. 

1. Ensure your Virtual Environment is active:
```bash
source venv/bin/activate
```
2. Start the Flask Backend Server:
```bash
python app.py
```
3. Open your browser to access the beautiful User Interface! 
👉 **http://127.0.0.1:8000** 

---

## 🧠 Application Architecture
1. **Submit:** User queries "Machine Learning Engineer" and states their current skills.
2. **Fetch:** `RemotiveScraper` pulls in real, remote job descriptions.
3. **Backup:** `CloudStorage` serializes and uploads data to Supabase `/indeed-jobs` bucket.
4. **Vectorize:** `Langchain` slices documents into dense 1000-character chunks; embedded iteratively with `mistral-embed`.
5. **Analyze:** Mistral cross-examines context chunks extracted dynamically by `ChromaDB` against the user's background. 
6. **Serve:** A detailed gap analysis, curated timeline, and relevant courses are projected beautifully to the dashboard in real-time.

---

## 📝 License
MIT License. Feel free to fork and build upon this structure for your own RAG pipelines!
