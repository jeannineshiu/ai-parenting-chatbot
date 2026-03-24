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
