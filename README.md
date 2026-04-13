# SkillGap AI 🎯

*An advanced, real-time Retrieval-Augmented Generation (RAG) platform that eliminates the guesswork from job hunting by dynamically analyzing live market demands against your personal skill set.*

![Screenshot](/frontend/styles.css) *(Add a screenshot here later!)*

SkillGap AI is an intelligent career advisor and pipeline engine. Typical course roadmaps are static; SkillGap AI is entirely dynamic. By aggregating live job descriptions through asynchronous multi-board web scraping, processing them through a Hybrid Search vector architecture, and piping the results into Mistral's LLM, the system constructs a precise, immediately actionable learning plan specifically tailored to you.

---

## 🚀 Key Features

* **Multi-Board Live Data Ingestion:** Synchronously searches across 5 major remote-job API registries (Remotive, Arbeitnow, Adzuna, WeWorkRemotely, Himalayas) to aggregate a high-fidelity dataset of real-time job openings in seconds.
* **Semantic Chunking:** Custom LangChain Document Builder logically segments job descriptions by detecting textual headers (e.g., "Requirements", "What you'll do"), and enriches vector points dynamically with Metadata (Seniority, Salary, Remote tier) via Regex.
* **Intelligent Caching:** PostgreSQL logging safely caches exact user queries in Supabase; if a query was run within the last 24 hours, it bypasses scraping to save computational overhead.
* **Hybrid Search RAG (RRF):** Fuses standard Dense vector logic (Mistral-based cosine similarity) via ChromaDB with Sparse keyword math (BM25Okapi) to catch specific tech-stack terminologies. Analyzes retrieved chunks using Reciprocal Rank Fusion (RRF).
* **Production-Ready Deployment:** Packaged securely for Render.com leveraging Gunicorn's `gthread` async worker architecture explicitly tuned for AI memory-constrained nodes via lazy-loading modules.

---

## 🛠 Tech Stack

* **Language:** Python 3.11+
* **LLM & Embeddings:** Mistral AI (`mistral-small-latest` & `mistral-embed`)
* **Retrieval Tech:** ChromaDB (Dense Vectors), `rank_bm25` (Sparse Vectors)
* **Frameworks:** LangChain, Flask API Gateway, Gunicorn Web Server
* **Cloud Infrastructure:** Render, Supabase Storage (S3-Lake) & PostgreSQL
* **Web Scraping:** BeautifulSoup4, Requests
* **Frontend:** Vanilla HTML/JS styled distinctly with Notch/Linear-inspired sleek dark modes

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
Create a `.env` file in the root directory and securely add required API Keys. 
*(Sign-up for free tier API keys at Mistral, Supabase, and Adzuna).*
```env
# Mistral API (For embeddings and LLM generation)
MISTRAL_API_KEY=your_mistral_api_key

# Supabase Configurations (Logs + Raw Json backups)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key

# Job APIs
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key

# Local/Cloud Vector store routing
CHROMA_PERSIST_DIR=./chroma_db
```

---

## 🏃‍♂️ Running the Application Locally

SkillGap AI operates using a **Flask Wrapper API** connected seamlessly to a pristine HTML frontend. 

1. Start the backend:
```bash
python app.py
```
2. Open your browser to access the sleek User Interface! 
👉 **http://127.0.0.1:8000** 

---

## 🚢 Deploying to Render

This application includes a custom `render.yaml` configuration template tuned precisely for Python Machine Learning deployments on limited RAM constraints. 

1. Connect your GitHub repository to Render using **Blueprint Deploy** or create a web service.
2. In the Render Dashboard, ensure the settings correspond exactly:
   * **Build Command:** `pip install -r requirements.txt && python download_models.py`
   * **Start Command:** `gunicorn app:app --worker-class gthread --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
   * **Mount Path:** Deploy the Persistent Disk at `/var/data` (to prevent Render shadow-clone overlapping).

*(Note: PyTorch multi-threading flags like `OMP_NUM_THREADS` are aggressively locked to 1 in the yaml to prevent OOM memory explosions on Free/Hobby tiers).*

---

## 🧠 System Architecture Workflow
1. **Query & Lock:** User submits Query (e.g. "Data Engineer") + Current Skills Context. Supabase evaluates freshness parameters.
2. **Scraper Registry:** Spins up 5 simultaneous board fetch routines, scrubs HTML tags, and deduplicates identical job links into a cohesive `JobListing` payload model.
3. **Semantic Splitter:** Maps descriptions structurally into targeted LangChain Documents and injects salary/location metadata.
4. **Vector Storage:** Data is natively ingested to ChromaDB on the deployed local disk cache. 
5. **Hybrid Retrieval Synthesis:** Dense Mistral embedding matching performs a cross-join against lexical BM25 matching arrays.
6. **LLM Generative Insight:** The absolute top candidates are pipelined into Mistral with strict Generation instructions.
7. **Render Display:** Outputs structured JSON consisting of Top Matching Job Roles, Global In-Demand Tech Stacks, and heavily personalized Action Plans directly back to the DOM frontend interface dynamically.

---

## 🔮 Future Architecture (V2)
- Re-activate **MS-MARCO CrossEncoder Reranking** by migrating heavy inferences to serverless GPU inference endpoints (Modal/Bedrock) to bypass Server memory limitations securely.
- Integrate an independent async tasks execution queue (via Celery/RabbitMQ).
- Self-evaluating generation logic utilizing Information Retrieval matrices internally tracking Precision@K & MRR scoring automatically logged via `evaluator/retrival_eval.py`.

---

## 📝 License
MIT License. Feel free to copy, mutate, and construct your own intelligent RAG pipelines!
