from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_documents, semantic_search, generate_answer

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

docs = load_documents("../data")
print(f"Loaded {len(docs)} documents")

chat_history = []

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):

    global chat_history

    results = semantic_search(query.question, docs)

    context = "\n\n".join([doc["content"][:500] for doc in results])

    chat_history.append({"role": "user", "content": query.question})

    messages = [
    {
            "role": "system",
            "content": f"""
    You are a helpful parenting assistant.

    Use the following context to answer the user's question.
    Be clear, concise, and helpful. Do not copy text directly.

    Context:
    {context}
    """
        },
        *chat_history[-6:]
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content

    chat_history.append({"role": "assistant", "content": answer})

    return {"answer": answer}