# AI Chatbot

A Streamlit chat app that talks to many LLMs through [OpenRouter](https://openrouter.ai), with streaming replies,
a model picker (including free models), an editable system prompt and a max-reply-tokens slider.

## Required secret

| Name | Where to get it |
|------|-----------------|
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys |

Never commit the key. `.streamlit/secrets.toml` is git-ignored.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p .streamlit
echo 'OPENROUTER_API_KEY = "your-key-here"' > .streamlit/secrets.toml
streamlit run app.py
```

Open http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io click **Create app**, pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**, paste: `OPENROUTER_API_KEY = "your-key-here"`.
4. Deploy. (Dependencies come from `requirements.txt`; `streamlit_requirements.txt` is kept identical for older configs.)

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q && ruff check .
```

Tests mock OpenRouter, so no key or network is needed.

## Files

- `app.py` – Streamlit UI
- `chat_api.py` – OpenRouter streaming client, model list, history cap
- `tests/` – pytest suite
