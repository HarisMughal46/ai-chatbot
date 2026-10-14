"""
AI Chatbot — Streamlit app
Calls OpenRouter directly with streaming replies.
Run: streamlit run app.py   (needs OPENROUTER_API_KEY in Streamlit secrets)
"""

import json
from datetime import datetime

import streamlit as st

from chat_api import MAX_HISTORY_MESSAGES, MODELS, ChatError, stream_chat

PERSONAS = {
    "Helpful assistant": (
        "You are a helpful, knowledgeable, and friendly AI assistant. "
        "You give clear, accurate, and concise answers. "
        "When you are unsure about something, you say so."
    ),
    "Senior software engineer": (
        "You are a senior software engineer. Give precise, practical answers with working code "
        "in fenced code blocks, and explain trade-offs briefly."
    ),
    "Writing editor": (
        "You are a professional editor. Improve clarity, grammar and tone while keeping the author's voice."
    ),
    "Custom": "",
}
DEFAULT_PERSONA = "Helpful assistant"
SUGGESTIONS = [
    ("💡", "Explain AI in simple terms"),
    ("🐍", "Write a Python function to merge two sorted lists"),
    ("📝", "Give me 5 productivity tips"),
    ("✉️", "Draft a polite follow-up email"),
]

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")


# ── Secrets ────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:  # no secrets.toml at all
        key = None
    return key.strip() if isinstance(key, str) and key.strip() else None


OPENROUTER_API_KEY = get_api_key()

# ── Styling ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 820px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid #232838; }
[data-testid="stSidebar"] h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: .08em;
    color: #8B90A0; margin: 0.6rem 0 0.2rem; }
[data-testid="stChatMessage"] { border-radius: 14px; padding: 0.75rem 1rem; margin-bottom: 0.6rem;
    border: 1px solid #232838; background: #151924; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, #2A2660 0%, #221F4A 100%); border-color: #3A3480; }
[data-testid="stChatMessage"] pre { border-radius: 10px; }
[data-testid="stChatMessage"] code { font-size: 0.86rem; }
.app-header { text-align: center; padding: 0.5rem 0 1.25rem; }
.app-header .logo { width: 56px; height: 56px; margin: 0 auto 0.6rem; border-radius: 16px; display: flex;
    align-items: center; justify-content: center; font-size: 28px;
    background: linear-gradient(135deg, #7C6CFF, #4F8CFF); box-shadow: 0 8px 24px rgba(124,108,255,.35); }
.app-header h1 { font-size: 1.7rem; font-weight: 700; margin: 0; padding: 0; }
.app-header p { color: #8B90A0; font-size: 0.92rem; margin: 0.25rem 0 0; }
.model-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem;
    background: #1D2230; border: 1px solid #2C3246; color: #B8BCCB; margin-top: 0.5rem; }
.stButton > button { border-radius: 10px; font-weight: 500; }
@media (max-width: 640px) {
  .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
  .app-header h1 { font-size: 1.35rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────
DEFAULTS = {
    "messages": [],
    "total_tokens": 0,
    "total_requests": 0,
    "selected_model": next(iter(MODELS)),
    "persona": DEFAULT_PERSONA,
    "system_prompt": PERSONAS[DEFAULT_PERSONA],
    "max_tokens": 1024,
    "temperature": 0.7,
    "last_error": None,
    "pending": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, list) else v


def reset_chat():
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.total_requests = 0
    st.session_state.last_error = None


def on_persona_change():
    p = st.session_state.persona
    if p != "Custom":
        st.session_state.system_prompt = PERSONAS[p]


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Chatbot")
    st.caption("Multi-model chat powered by OpenRouter")

    st.markdown("### Model")
    st.selectbox("Model", options=list(MODELS.keys()), key="selected_model", label_visibility="collapsed")
    model_id = MODELS[st.session_state.selected_model]
    st.caption(f"`{model_id}`")

    st.markdown("### Personality")
    st.selectbox(
        "Preset",
        options=list(PERSONAS.keys()),
        key="persona",
        on_change=on_persona_change,
        label_visibility="collapsed",
    )
    st.text_area(
        "System prompt",
        key="system_prompt",
        height=110,
        label_visibility="collapsed",
        help="Tells the AI how to behave.",
        placeholder="You are a helpful assistant…",
    )

    with st.expander("⚙️ Generation settings"):
        st.slider("Max reply tokens", 128, 4096, step=128, key="max_tokens")
        st.slider("Temperature", 0.0, 1.5, step=0.1, key="temperature", help="Lower = focused, higher = creative.")
        st.caption(f"Only the last {MAX_HISTORY_MESSAGES} messages are sent to the model.")

    st.markdown("### Session")
    c1, c2, c3 = st.columns(3)
    c1.metric("Msgs", len(st.session_state.messages))
    c2.metric("Tokens", f"{st.session_state.total_tokens:,}")
    c3.metric("Calls", st.session_state.total_requests)

    b1, b2 = st.columns(2)
    b1.button(
        "🗑️ Clear",
        key="clear_btn",
        on_click=reset_chat,
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    export = json.dumps(
        {"model": model_id, "system_prompt": st.session_state.system_prompt, "messages": st.session_state.messages},
        indent=2,
        ensure_ascii=False,
    )
    b2.download_button(
        "⬇️ Export",
        export,
        file_name=f"chat-{datetime.now():%Y%m%d-%H%M}.json",
        mime="application/json",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    st.divider()
    st.caption("Built with Streamlit + OpenRouter")

# ── Header ─────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="app-header"><div class="logo">🤖</div><h1>AI Chatbot</h1>
<p>Ask anything — I remember the conversation.</p>
<span class="model-pill">{st.session_state.selected_model}</span></div>""",
    unsafe_allow_html=True,
)

if not OPENROUTER_API_KEY:
    st.warning(
        "**OPENROUTER_API_KEY is not set.**\n\n"
        'Locally: add `OPENROUTER_API_KEY = "..."` to `.streamlit/secrets.toml`.  \n'
        "Streamlit Cloud: App → Settings → Secrets. Get a key at https://openrouter.ai/keys"
    )
    st.stop()

# ── Empty state ────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.pending:
    cols = st.columns(2)
    for i, (icon, text) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {text}", key=f"chip_{i}", use_container_width=True):
            st.session_state.pending = text
            st.rerun()

# ── History ────────────────────────────────────────────────────────
AVATAR = {"user": "👤", "assistant": "🤖"}
msgs = st.session_state.messages
for msg in msgs:
    with st.chat_message(msg["role"], avatar=AVATAR[msg["role"]]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.popover("📋 Copy", use_container_width=False):
                st.code(msg["content"], language="markdown", wrap_lines=True)

can_regen = msgs and msgs[-1]["role"] == "assistant" and not st.session_state.pending
if can_regen and st.button("🔄 Regenerate last reply", key="regen_btn"):
    msgs.pop()
    st.session_state.pending = "__regen__"
    st.rerun()

# ── Input & generation ─────────────────────────────────────────────
user_text = st.session_state.pending
st.session_state.pending = None
chat_input = st.chat_input("Message the AI…")
if chat_input:
    user_text = chat_input

if st.session_state.last_error and not user_text:
    st.error(st.session_state.last_error)

if user_text:
    st.session_state.last_error = None
    added_user = False
    if user_text != "__regen__":
        msgs.append({"role": "user", "content": user_text})
        added_user = True
        with st.chat_message("user", avatar=AVATAR["user"]):
            st.markdown(user_text)

    usage: dict = {}
    reply = None
    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        try:
            reply = st.write_stream(
                stream_chat(
                    OPENROUTER_API_KEY,
                    msgs,
                    model_id,
                    st.session_state.system_prompt,
                    max_tokens=st.session_state.max_tokens,
                    temperature=st.session_state.temperature,
                    usage=usage,
                )
            )
        except ChatError as e:
            st.session_state.last_error = f"⚠️ **Error:** {e}"
        except Exception as e:  # unexpected
            st.session_state.last_error = f"⚠️ **Unexpected error:** {e}"

    if isinstance(reply, str) and reply.strip():
        msgs.append({"role": "assistant", "content": reply})
        st.session_state.total_tokens += usage.get("total_tokens", 0)
        st.session_state.total_requests += 1
        st.rerun()  # refresh sidebar stats; only on success
    else:
        if added_user:
            msgs.pop()  # let the user retry; error stays visible below
        if not st.session_state.last_error:
            st.session_state.last_error = "⚠️ **Error:** The model returned an empty reply."
        st.error(st.session_state.last_error)
