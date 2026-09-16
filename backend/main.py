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

docs = load_documents()
print(f"Loaded {len(docs)} documents")

# One history per session so concurrent users never see each other's messages.
# Each history holds the full OpenAI message sequence (user / assistant /
# assistant+tool_calls / tool) so that IDs returned by tools stay available
# across turns of a multi-step booking.
chat_histories: dict[str, list] = {}

# Number of most recent user turns to send back to the model
MAX_HISTORY_TURNS = 3

class Query(BaseModel):
    question: str
    session_id: str = "default"


def recent_turns(history: list, max_turns: int) -> list:
    """Return the last `max_turns` user turns, cutting only at a user message
    so an assistant tool_calls message is never separated from its tool results."""
    user_indexes = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_indexes) <= max_turns:
        return history
    return history[user_indexes[-max_turns]:]

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

# ------------- Chat Pipeline -------------

# Characters of each retrieved chunk passed to the model (None = whole chunk).
# Truncating to 500 cut off most chunks (median ~1,000 chars); RAGAS showed whole
# chunks raise faithfulness 0.68 -> 0.88 and context recall 0.59 -> 0.94.
CONTEXT_CHARS = None


def retrieve(question: str) -> tuple[list, list[str]]:
    """Return the top retrieved documents and the context strings the model sees."""
    results = semantic_search(question, docs)
    contexts = [doc["content"][:CONTEXT_CHARS] for doc in results]
    return results, contexts


def build_system_message(contexts: list[str]) -> dict:
    context = "\n\n".join(contexts)
    return {
        "role": "system",
        "content": f"""Du bist ein einfühlsamer Eltern-Assistent von ElternLeben.de.

Beantworte Fragen auf Basis des folgenden Kontexts. Sei klar, hilfreich und unterstützend.
Kopiere den Text nicht direkt.

Wenn der Nutzer ein Webinar sucht oder buchen möchte, nutze get_webinars und register_for_webinar.
Wenn der Nutzer eine persönliche Beratung möchte, nutze get_available_experts, get_expert_slots und book_consultation.
Frage nach fehlenden Informationen (Name, E-Mail), bevor du eine Buchung durchführst.

Kontext:
{context}""",
    }


def run_chat(question: str, history: list) -> tuple[str, list, list[str]]:
    """Answer `question` using RAG + tool calls, appending all messages to `history`.

    Shared by the /chat endpoint and the RAGAS evaluation (eval/run_ragas.py),
    so both exercise exactly the same pipeline.
    Returns (answer, retrieved documents, context strings given to the model).
    """
    results, contexts = retrieve(question)
    history.append({"role": "user", "content": question})
    system_message = build_system_message(contexts)

    def complete():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, *recent_turns(history, MAX_HISTORY_TURNS)],
            tools=tools,
            temperature=0.3,
        )

    response = complete()

    # Handle tool call loop. Every intermediate message is stored in the session
    # history so tool results (expert IDs, webinar IDs, ...) survive to later turns.
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            try:
                result = call_tool(tc.function.name, args)
            except (http_requests.RequestException, ValueError) as e:
                # Let the model tell the user the service is unavailable instead of crashing
                result = {"error": f"Service nicht erreichbar: {e}"}
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        response = complete()

    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer, results, contexts

# ------------- Chat Endpoint -------------

@app.post("/chat")
def chat(query: Query):
    history = chat_histories.setdefault(query.session_id, [])
    answer, results, _ = run_chat(query.question, history)

    sources = list(dict.fromkeys(
        doc["url"] for doc in results if doc.get("url")
    ))

    return {"answer": answer, "sources": sources}
