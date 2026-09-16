import uuid

import gradio as gr
import requests

# Backend API endpoint inside the Docker network
API_URL = "http://backend:8000/chat"

def chat(message, history, session_id):
    """
    Handles user interaction by posting the question to the backend
    and formatting the response with sources.
    """
    # One session ID per browser tab so the backend keeps histories separate
    session_id = session_id or str(uuid.uuid4())

    try:
        # Requesting answer from backend with a 120s timeout (multi-tool flows can be slow)
        response = requests.post(
            API_URL,
            json={"question": message, "session_id": session_id},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "No answer received from the assistant.")
        sources = data.get("sources", [])

        # Format sources into a Markdown list if they exist
        if sources:
            sources_text = "\n".join([
                f"- [{s.rstrip('/').split('/')[-1]}]({s})"
                for s in sources
            ])
            full_response = f"{answer}\n\n📚 **Sources:**\n{sources_text}"
        else:
            full_response = answer

    except Exception as e:
        # Catch connection errors or API issues
        full_response = f"⚠️ **Connection Error:** Backend is unreachable. ({str(e)})"

    # Append interaction to history (list of dicts)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})

    # Return empty string to clear input, updated history for display, and the session ID
    return "", history, session_id


with gr.Blocks(title="AI Parenting Assistant") as demo:
    gr.Markdown("# 👶 AI Parenting Assistant")

    # gr.State is per browser session, so each tab gets its own conversation
    session_id = gr.State(value=None)

    # Removed 'show_copy_button' as it's no longer a valid argument in Gradio 5.x
    chatbot = gr.Chatbot(label="Conversation")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about parenting...",
            show_label=False,
            scale=9
        )
        submit_btn = gr.Button("Send", scale=1)

    # Event handlers for submitting via Enter or Button click
    msg.submit(chat, [msg, chatbot, session_id], [msg, chatbot, session_id])
    submit_btn.click(chat, [msg, chatbot, session_id], [msg, chatbot, session_id])

if __name__ == "__main__":
    # Launching on 0.0.0.0 to ensure Docker port forwarding works
    demo.launch(server_name="0.0.0.0", share=False)