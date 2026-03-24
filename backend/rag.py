import os

from openai import OpenAI
import numpy as np

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

print("API KEY LOADED:", api_key[:5])

query_cache = {}

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

import json

CACHE_FILE = os.path.join(os.path.dirname(__file__), "embedding_cache.json")

def clean_text(text):
    lines = text.split("\n")
    
    # 移除 markdown metadata（--- 開頭）
    cleaned = []
    skip = False

    for line in lines:
        if line.strip() == "---":
            skip = not skip
            continue
        if not skip:
            cleaned.append(line)

    text = "\n".join(cleaned)

    # 移除圖片 markdown
    text = text.replace("![", "").replace("](", "(")

    return text.strip()

def chunk_text(text, chunk_size=150):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

def load_documents(folder_path="data"):
    docs = []

    # 如果 cache 存在 → 直接讀
    if os.path.exists(CACHE_FILE):
        print("Loading embeddings from cache...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("No cache found, computing embeddings...")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    content = clean_text(content)

                    chunks = chunk_text(content)

                    for chunk in chunks:
                        #embedding = get_embedding(chunk)
                        embedding = np.array(get_embedding(chunk))

                        docs.append({
                            "content": chunk,
                            "embedding": embedding,
                            "source": file_path
                        })

    # 存 cache
    with open(CACHE_FILE, "w") as f:
        json.dump(docs, f, separators=(",", ":"))

    return docs


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return np.dot(a, b) / denom


def semantic_search(query, docs):
    if query in query_cache:
        query_embedding = query_cache[query]
    else:
        #query_embedding = get_embedding(query)
        query_embedding = np.array(get_embedding(query))
        query_cache[query] = query_embedding

    scored = []
    for doc in docs:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored[:5]]  # 多拿一點

def generate_answer(query, context_docs):
    context = "\n\n".join([
        f"[Source {i+1}]\n{doc['content'][:300]}"
        for i, doc in enumerate(context_docs)
    ])

    sources = list(set(doc["source"] for doc in context_docs))

    prompt = f"""
    You are a helpful parenting assistant.

    Use ONLY the provided context to answer the question.

    Guidelines:
    - Answer in German
    - Be clear, structured, and supportive
    - Give practical advice
    - Do NOT copy text directly
    - If helpful, ask one follow-up question

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful parenting assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }
