"""
AI Chatbot — Streamlit App
Calls OpenRouter directly (no separate FastAPI server needed).
Deploy free at: https://streamlit.io/cloud
"""

import streamlit as st
import requests
import json

# ══════════════════════════════════════════════════════════════════
#  CONFIG — your OpenRouter key is already set here
# ══════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "GPT-4o Mini ⚡ (Fast)"        : "openai/gpt-4o-mini",
    "GPT-4o 🧠 (Most Capable)"     : "openai/gpt-4o",
    "Claude 3.5 Sonnet 🎯"         : "anthropic/claude-3-5-sonnet",
    "Llama 3.1 8B 🆓 (Free)"       : "meta-llama/llama-3.1-8b-instruct:free",
    "Gemini Flash ✨"               : "google/gemini-flash-1.5",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and friendly AI assistant. "
    "You give clear, accurate, and concise answers. "
    "When you are unsure about something, you say so."
)

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════
#  CUSTOM CSS  — dark theme matching original chatbot
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14 !important;
    color: #e8eaf2 !important;
}
[data-testid="stSidebar"] {
    background-color: #161920 !important;
    border-right: 1px solid #2a2f42;
}
[data-testid="stSidebar"] * { color: #e8eaf2 !important; }

/* ── Hide Streamlit default header/footer ────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: #1e2130 !important;
    border: 1px solid #2a2f42 !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    color: #e8eaf2 !important;
}
[data-testid="stChatMessage"] p { color: #e8eaf2 !important; }

/* ── User message ────────────────────────────────────────── */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #2d2b6b !important;
}

/* ── Input box ───────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background-color: #1e2130 !important;
    color: #e8eaf2 !important;
    border: 1px solid #2a2f42 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
}

/* ── Selectbox ───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: #1e2130 !important;
    color: #e8eaf2 !important;
    border: 1px solid #2a2f42 !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background-color: #6c63ff !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: #5a52d5 !important;
    transform: scale(1.02);
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #1e2130;
    border: 1px solid #2a2f42;
    border-radius: 10px;
    padding: 10px 16px;
}
[data-testid="stMetricValue"] { color: #6c63ff !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #2a2f42 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE  — persists across re-runs within one session
# ══════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": ..., "content": ...}

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "total_requests" not in st.session_state:
    st.session_state.total_requests = 0

if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(MODELS.keys())[0]

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 AI Chatbot")
    st.markdown("*Powered by OpenRouter*")
    st.divider()

    # Model picker
    st.markdown("#### 🧠 Choose AI Model")
    selected_label = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(st.session_state.selected_model),
        label_visibility="collapsed",
    )
    st.session_state.selected_model = selected_label
    model_id = MODELS[selected_label]
    st.caption(f"`{model_id}`")

    st.divider()

    # System prompt editor
    st.markdown("#### ⚙️ AI Personality")
    new_system = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=120,
        label_visibility="collapsed",
        help="This tells the AI how to behave. Change it to make the AI a specialist.",
        placeholder="You are a helpful assistant...",
    )
    st.session_state.system_prompt = new_system

    st.divider()

    # Stats
    st.markdown("#### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Requests sent", st.session_state.total_requests)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_requests = 0
        st.rerun()

    st.divider()
    st.caption("Built with FastAPI + OpenRouter")

# ══════════════════════════════════════════════════════════════════
#  MAIN AREA — Header
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
  <div style="font-size:48px">🤖</div>
  <h1 style="color:#e8eaf2; margin:8px 0 4px 0;">AI Chatbot</h1>
  <p style="color:#6b7280; font-size:14px;">Ask me anything — I remember your full conversation</p>
</div>
""", unsafe_allow_html=True)

# ── Welcome suggestions (shown only when chat is empty) ───────────
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    suggestions = [
        "Explain AI in simple terms",
        "Write a short poem about coding",
        "What is Python used for?",
        "Give me 5 productivity tips",
    ]
    for i, chip in enumerate(suggestions):
        col = c1 if i % 2 == 0 else c2
        with col:
            if st.button(chip, use_container_width=True, key=f"chip_{i}"):
                st.session_state._quick_send = chip
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  DISPLAY CHAT HISTORY
# ══════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ══════════════════════════════════════════════════════════════════
#  OPENROUTER API CALL
# ══════════════════════════════════════════════════════════════════

def ask_openrouter(messages: list, model: str, system_prompt: str) -> tuple[str, int]:
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://streamlit.io",
        "X-Title":       "AI Chatbot",
    }
    payload = {
        "model":       model,
        "messages":    full_messages,
        "temperature": 0.7,
        "max_tokens":  2048,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to OpenRouter. Check internet.")

    if resp.status_code != 200:
        try:
            err_json = resp.json()
            err_msg  = err_json.get("error", {}).get("message", str(err_json))
        except Exception:
            err_msg = resp.text
        raise Exception(f"Status {resp.status_code}: {err_msg} — Your API key may be revoked. Go to openrouter.ai to generate a new one, then update Streamlit Secrets.")

    data = resp.json()
    if "choices" not in data or len(data["choices"]) == 0:
        raise Exception(f"Unexpected response: {data}")

    reply       = data["choices"][0]["message"]["content"]
    tokens_used = data.get("usage", {}).get("total_tokens", 0)
    return reply, tokens_used

# ══════════════════════════════════════════════════════════════════
#  HANDLE INPUT
# ══════════════════════════════════════════════════════════════════

# Handle quick-send from suggestion chips
user_text = st.session_state.pop("_quick_send", None)

# Handle normal chat input
chat_input = st.chat_input("Type your message… (Enter to send)")
if chat_input:
    user_text = chat_input

if user_text:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)

    # Call API and stream the response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                reply, tokens = ask_openrouter(
                    st.session_state.messages,
                    MODELS[st.session_state.selected_model],
                    st.session_state.system_prompt,
                )
                st.markdown(reply)

                # Save assistant reply to history
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.total_tokens   += tokens
                st.session_state.total_requests += 1

            except Exception as e:
                err_msg = f"⚠️ **Error:** {str(e)}\n\nMake sure your API key is valid."
                st.error(err_msg)
                # Remove the user message we just added so they can retry
                st.session_state.messages.pop()

    st.rerun()
