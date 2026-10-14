"""
Lumen — a clean multi-model AI chat built on Streamlit + OpenRouter.
Run: streamlit run app.py   (needs OPENROUTER_API_KEY in Streamlit secrets)
"""

import html
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from chat_api import MAX_HISTORY_MESSAGES, MODELS, ChatError, stream_chat

APP_NAME = "Lumen"
FAVICON = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%23A99BFF'/>"
    "<stop offset='1' stop-color='%236D5DF0'/></linearGradient></defs>"
    "<rect width='64' height='64' rx='16' fill='url(%23g)'/>"
    "<path d='M32 14l4.5 13.5L50 32l-13.5 4.5L32 50l-4.5-13.5L14 32l13.5-4.5z' fill='white'/></svg>"
)


def logo(size: int = 26) -> str:
    style = f"display:block;border-radius:{size // 4}px"
    return f'<img src="{FAVICON}" width="{size}" height="{size}" alt="" style="{style}">'


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
    ("Explain a concept", "Explain AI in simple terms"),
    ("Write code", "Write a Python function to merge two sorted lists"),
    ("Get advice", "Give me 5 productivity tips"),
    ("Draft a message", "Draft a polite follow-up email"),
]

st.set_page_config(page_title=APP_NAME, page_icon="✦", layout="centered", initial_sidebar_state="auto")
# Streamlit's page_icon can't take a data URI, so inject the SVG favicon too.
st.html(f'<link rel="icon" href="{FAVICON}">')


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:  # no secrets.toml at all
        key = None
    return key.strip() if isinstance(key, str) and key.strip() else None


OPENROUTER_API_KEY = get_api_key()

# ── Styling ────────────────────────────────────────────────────────
st.html(f"<style>{(Path(__file__).parent / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")

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
    st.html(f'<div class="side-brand">{logo()}<span>{APP_NAME}</span></div>')
    with st.container(key="newchat"):
        st.button("＋  New chat", key="clear_btn", on_click=reset_chat, use_container_width=True)

    st.html('<div class="side-label">Model</div>')
    st.selectbox("Model", options=list(MODELS.keys()), key="selected_model", label_visibility="collapsed")
    model_id = MODELS[st.session_state.selected_model]

    st.html('<div class="side-label">Preferences</div>')
    with st.expander("Settings", icon=":material/tune:"):
        st.selectbox("Persona", options=list(PERSONAS.keys()), key="persona", on_change=on_persona_change)
        st.text_area("System prompt", key="system_prompt", height=110, placeholder="You are a helpful assistant…")
        st.slider("Temperature", 0.0, 1.5, step=0.1, key="temperature", help="Lower = focused, higher = creative.")
        st.slider("Max reply tokens", 128, 4096, step=128, key="max_tokens")
        st.caption(f"Context: last {MAX_HISTORY_MESSAGES} messages are sent.")

    st.html('<div class="side-label">Session</div>')
    st.html(
        f'<div class="stats"><div><b>{len(st.session_state.messages)}</b><span>Messages</span></div>'
        f"<div><b>{st.session_state.total_tokens:,}</b><span>Tokens</span></div>"
        f"<div><b>{st.session_state.total_requests}</b><span>Calls</span></div></div>"
    )
    export = json.dumps(
        {"model": model_id, "system_prompt": st.session_state.system_prompt, "messages": st.session_state.messages},
        indent=2,
        ensure_ascii=False,
    )
    st.download_button(
        "Export chat",
        export,
        file_name=f"lumen-chat-{datetime.now():%Y%m%d-%H%M}.json",
        mime="application/json",
        icon=":material/download:",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    st.html(f'<div class="side-foot">{APP_NAME} · Streamlit + OpenRouter</div>')

# ── Top bar ────────────────────────────────────────────────────────
st.html(
    f'<div class="topbar"><div class="brand">{logo()}<span>{APP_NAME}</span></div>'
    f'<div class="model">{html.escape(model_id)}</div></div>'
)

if not OPENROUTER_API_KEY:
    st.warning(
        "**OPENROUTER_API_KEY is not set.**\n\n"
        'Locally: add `OPENROUTER_API_KEY = "..."` to `.streamlit/secrets.toml`.  \n'
        "Streamlit Cloud: App → Settings → Secrets. Get a key at https://openrouter.ai/keys"
    )
    st.stop()

msgs = st.session_state.messages

# ── Empty state ────────────────────────────────────────────────────
if not msgs and not st.session_state.pending:
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.html(
        f'<div class="hero"><div class="mark">{logo(48)}</div><h1>{greeting}</h1><p>How can I help you today?</p></div>'
    )
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, (title, text) in enumerate(SUGGESTIONS):
            if cols[i % 2].button(f"**{title}**  \n{text}", key=f"chip_{i}", use_container_width=True):
                st.session_state.pending = text
                st.rerun()

# ── History ────────────────────────────────────────────────────────
AVATAR = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}
for i, msg in enumerate(msgs):
    with st.chat_message(msg["role"], avatar=AVATAR[msg["role"]]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            is_last = i == len(msgs) - 1
            with st.container(key=f"actions_{i}" if not is_last else "actions"):
                a1, a2, _ = st.columns([1.1, 1.6, 4], gap="small")
                with a1.popover("Copy", icon=":material/content_copy:"):
                    st.code(msg["content"], language="markdown", wrap_lines=True)
                if (
                    is_last
                    and not st.session_state.pending
                    and a2.button("Regenerate", key="regen_btn", icon=":material/refresh:")
                ):
                    msgs.pop()
                    st.session_state.pending = "__regen__"
                    st.rerun()

# ── Input & generation ─────────────────────────────────────────────
user_text = st.session_state.pending
st.session_state.pending = None
chat_input = st.chat_input(f"Message {APP_NAME}…")
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
            st.session_state.last_error = f"**Error:** {e}"
        except Exception as e:  # unexpected
            st.session_state.last_error = f"**Unexpected error:** {e}"

    if isinstance(reply, str) and reply.strip():
        msgs.append({"role": "assistant", "content": reply})
        st.session_state.total_tokens += usage.get("total_tokens", 0)
        st.session_state.total_requests += 1
        st.rerun()  # refresh sidebar stats; only on success
    else:
        if added_user:
            msgs.pop()  # let the user retry; error stays visible below
        if not st.session_state.last_error:
            st.session_state.last_error = "**Error:** The model returned an empty reply."
        st.error(st.session_state.last_error)
