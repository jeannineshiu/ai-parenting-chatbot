# 👶 AI Parenting Assistant: Empowering Families with ElternLeben.de

This project was developed as a solution for the **"AI for Impact" Hackathon**, focusing on transforming digital family support in Germany. It leverages the expert-vetted knowledge base of **ElternLeben.de** to provide parents with 24/7, judgment-free, and research-based guidance.

---

## 🌟 Mission & Impact

Parents often feel overwhelmed by conflicting online advice. Our goal is to bridge the gap between **professional expertise** and **instant accessibility**. By grounding a Large Language Model (LLM) in **750+ verified articles** from ElternLeben.de, we ensure that the AI's support is:

* ✅ **Evidence-Based:** Every answer is derived from specialist-curated content.
* ✅ **Socially Impactful:** Making high-quality counseling accessible to all families, regardless of background.
* ✅ **Transparent:** Directly citing the original expert articles to build trust.

---

## 🚀 Key Features

* **RAG-Powered Reliability:** Uses Retrieval-Augmented Generation to minimize hallucinations and prioritize expert knowledge.
* **Source Attribution:** Automatically provides clickable links to the original Markdown sources from the ElternLeben dataset.
* **Dockerized Microservices:** A professional-grade architecture featuring a **FastAPI** backend and a **Gradio 5** frontend.
* **German Language Support:** Optimized for the German parenting context and terminology.

---

## 🏗️ System Architecture

The application is split into two primary services communicating over a private Docker network:

1.  **Backend (FastAPI):** Orchestrates the RAG pipeline—document indexing, semantic search, and context-injected response generation.
2.  **Frontend (Gradio 5):** A modern, reactive UI that supports multi-turn conversations and handles API errors gracefully.



---

## 🧠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **LLM / Embeddings** | OpenAI GPT-4o / Azure OpenAI |
| **Orchestration** | Docker & Docker Compose |
| **Backend Framework** | FastAPI (Python 3.10+) |
| **Frontend UI** | Gradio 5.x |
| **Data Source** | ElternLeben.de Knowledge Base (750+ MD files) |

---

## 📂 Project Structure

```text
├── backend/            # RAG implementation & FastAPI endpoints
├── frontend/           # Gradio UI (optimized for Gradio 5.x)
├── data/               # 750+ Markdown articles from ElternLeben.de
├── .env.example        # Template for environment variables
└── docker-compose.yml  # Microservices orchestration

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ai-parenting-chatbot.git](https://github.com/your-username/ai-parenting-chatbot.git)
cd ai-parenting-chatbot

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your API key:
```bash
OPENAI_API_KEY=your_actual_key_here

### 3. Launch with Docker
```bash
docker compose up --build

### 4. Access the Assistant
* Frontend UI: http://localhost:7860

* API Documentation: http://localhost:8000/docs

## 📌 Technical Deep Dive: The RAG Pipeline
* Ingestion: 750+ Markdown files from the ElternLeben dataset are processed, cleaned, and partitioned into chunks.

* Retrieval: User queries are vectorized and matched against the knowledge base using semantic similarity.

* Augmentation: The most relevant expert content is injected into the LLM prompt as context.

* Generation: The model generates a response grounded strictly in the provided ElternLeben expertise to ensure safety and accuracy.

## 🛣️ Roadmap & Future Improvements

- [ ] **Vector Database:** Transition from in-memory search to **FAISS** or **ChromaDB** for enhanced scalability and performance as the knowledge base grows.
- [ ] **Hybrid Search:** Combine semantic search with **BM25 keyword matching** to better handle specific German medical, legal, and parenting terminology.
- [ ] **Evaluation:** Implement **Ragas** (RAG Assessment) to quantitatively measure "Faithfulness," "Answer Relevancy," and "Context Precision."
- [ ] **Cloud Deployment:** Professional production deployment via **Azure Container Apps** for high availability and automated scaling.

