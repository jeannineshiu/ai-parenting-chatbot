from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_documents, semantic_search

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import requests as http_requests

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# host.docker.internal resolves to the host machine from inside Docker on macOS
MOCK_API_URL = os.getenv("MOCK_API_URL", "http://host.docker.internal:8001")

app = FastAPI()

docs = load_documents("../data")
print(f"Loaded {len(docs)} documents")

chat_history = []

class Query(BaseModel):
    question: str

# ------------- Tool Definitions -------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_webinars",
            "description": (
                "Gibt eine Liste aller verfügbaren Webinare zurück. "
                "Verwende dies, wenn der Nutzer nach Webinaren oder Online-Seminaren fragt."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_for_webinar",
            "description": "Registriert einen Nutzer für ein bestimmtes Webinar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "webinar_id": {"type": "string", "description": "Die UUID des Webinars"},
                    "first_name": {"type": "string", "description": "Vorname des Nutzers"},
                    "last_name": {"type": "string", "description": "Nachname des Nutzers"},
                    "email": {"type": "string", "description": "E-Mail-Adresse des Nutzers"},
                },
                "required": ["webinar_id", "first_name", "email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_experts",
            "description": (
                "Gibt eine Liste aller verfügbaren Experten zurück. "
                "Verwende dies, wenn der Nutzer eine Beratung buchen möchte."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_expert_slots",
            "description": "Gibt die verfügbaren Zeitfenster eines bestimmten Experten zurück.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expert_id": {"type": "string", "description": "Die UUID des Experten"},
                },
                "required": ["expert_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_consultation",
            "description": "Bucht eine Beratung bei einem Experten für den Nutzer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expert_id": {"type": "string", "description": "Die UUID des Experten"},
                    "service": {
                        "type": "string",
                        "description": (
                            "Art der Beratung, z.B. schlafberatung, erziehungsberatung, "
                            "ernaehrungsberatung, erstberatung"
                        ),
                    },
                    "client_name": {"type": "string", "description": "Vollständiger Name des Klienten"},
                    "client_email": {"type": "string", "description": "E-Mail des Klienten"},
                    "client_phone": {"type": "string", "description": "Telefonnummer des Klienten"},
                },
                "required": ["expert_id", "service", "client_name"],
            },
        },
    },
]

# ------------- Tool Execution -------------

def call_tool(name: str, args: dict) -> dict:
    if name == "get_webinars":
        r = http_requests.get(f"{MOCK_API_URL}/webinars", timeout=5)
        return r.json()

    elif name == "register_for_webinar":
        r = http_requests.post(
            f"{MOCK_API_URL}/webinars/{args['webinar_id']}/registrants",
            json={
                "first_name": args["first_name"],
                "last_name": args.get("last_name", ""),
                "email": args["email"],
            },
            timeout=5,
        )
        return r.json()

    elif name == "get_available_experts":
        r = http_requests.get(f"{MOCK_API_URL}/experts/available", timeout=5)
        return r.json()

    elif name == "get_expert_slots":
        r = http_requests.get(
            f"{MOCK_API_URL}/experts/{args['expert_id']}/available-slots",
            timeout=5,
        )
        return r.json()

    elif name == "book_consultation":
        r = http_requests.post(
            f"{MOCK_API_URL}/bookings/new",
            json={
                "expert_id": args["expert_id"],
                "service": args["service"],
                "client_name": args["client_name"],
                "client_email": args.get("client_email"),
                "client_phone": args.get("client_phone"),
            },
            timeout=5,
        )
        return r.json()

    return {"error": f"Unknown tool: {name}"}

# ------------- Chat Endpoint -------------

@app.post("/chat")
def chat(query: Query):
    results = semantic_search(query.question, docs)
    context = "\n\n".join([doc["content"][:500] for doc in results])

    chat_history.append({"role": "user", "content": query.question})

    messages = [
        {
            "role": "system",
            "content": f"""Du bist ein einfühlsamer Eltern-Assistent von ElternLeben.de.

Beantworte Fragen auf Basis des folgenden Kontexts. Sei klar, hilfreich und unterstützend.
Kopiere den Text nicht direkt.

Wenn der Nutzer ein Webinar sucht oder buchen möchte, nutze get_webinars und register_for_webinar.
Wenn der Nutzer eine persönliche Beratung möchte, nutze get_available_experts, get_expert_slots und book_consultation.
Frage nach fehlenden Informationen (Name, E-Mail), bevor du eine Buchung durchführst.

Kontext:
{context}""",
        },
        *chat_history[-6:],
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        temperature=0.3,
    )

    # Handle tool call loop
    while response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        messages.append(response.choices[0].message)

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            result = call_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0.3,
        )

    answer = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": answer})

    sources = list(dict.fromkeys(
        doc["url"] for doc in results if doc.get("url")
    ))

    return {"answer": answer, "sources": sources}
