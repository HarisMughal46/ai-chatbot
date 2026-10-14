import json
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

import chat_api

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeResp:
    def __init__(self, status=200, chunks=None, body=None):
        self.status_code = status
        self._chunks = chunks or []
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=True):
        yield ": OPENROUTER PROCESSING"
        for c in self._chunks:
            yield "data: " + json.dumps(c)
            yield ""
        yield "data: [DONE]"


def sse(*parts, usage=None):
    out = [{"choices": [{"delta": {"content": p}}]} for p in parts]
    if usage:
        out.append({"choices": [], "usage": usage})
    return out


def make_app(key="dummy-key"):
    at = AppTest.from_file(APP, default_timeout=30)
    if key is not None:
        at.secrets["OPENROUTER_API_KEY"] = key
    return at


def test_stream_chat_parses_sse_and_usage():
    usage = {}
    with mock.patch(
        "chat_api.requests.post", return_value=FakeResp(chunks=sse("Hel", "lo", usage={"total_tokens": 7}))
    ) as p:
        out = "".join(
            chat_api.stream_chat(
                "k", [{"role": "user", "content": "hi"}], "openai/gpt-4o-mini", "sys", max_tokens=99, usage=usage
            )
        )
    assert out == "Hello"
    assert usage["total_tokens"] == 7
    payload = p.call_args.kwargs["json"]
    assert payload["stream"] is True and payload["max_tokens"] == 99
    assert payload["messages"][0]["role"] == "system"


def test_history_cap():
    msgs = [{"role": "user", "content": str(i)} for i in range(50)]
    assert len(chat_api.trim_history(msgs)) == chat_api.MAX_HISTORY_MESSAGES
    assert chat_api.trim_history(msgs)[-1]["content"] == "49"


def test_401_raises_chat_error():
    with (
        mock.patch("chat_api.requests.post", return_value=FakeResp(401, body={"error": {"message": "bad key"}})),
        pytest.raises(chat_api.ChatError, match=r"401.*bad key.*OPENROUTER_API_KEY"),
    ):
        list(chat_api.stream_chat("k", [], "m", "s"))


def test_missing_key_shows_friendly_message():
    at = make_app(key=None)
    at.run()
    assert not at.exception
    assert any("OPENROUTER_API_KEY is not set" in w.value for w in at.warning)


def test_chat_streams_reply_into_history():
    with mock.patch(
        "chat_api.requests.post", return_value=FakeResp(chunks=sse("Mock ", "reply", usage={"total_tokens": 12}))
    ):
        at = make_app()
        at.run()
        at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert at.session_state.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Mock reply"},
    ]
    assert at.session_state.total_tokens == 12


def test_401_error_stays_visible_and_message_removed():
    with mock.patch("chat_api.requests.post", return_value=FakeResp(401, body={"error": {"message": "bad key"}})):
        at = make_app()
        at.run()
        at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert at.session_state.messages == []
    assert any("401" in e.value for e in at.error)
    at.run()  # error persists across a rerun
    assert any("401" in e.value for e in at.error)


def test_model_ids_include_free():
    assert any(m.endswith(":free") or m == "openrouter/free" for m in chat_api.MODELS.values())


def _ok(*parts):
    return FakeResp(chunks=sse(*parts, usage={"total_tokens": 5}))


def test_suggestion_chip_sends():
    with mock.patch("chat_api.requests.post", return_value=_ok("chip answer")):
        at = make_app()
        at.run()
        at.button(key="chip_0").click().run()
    assert not at.exception
    assert at.session_state.messages[-1]["content"] == "chip answer"
    assert at.session_state.messages[0]["content"] == "Explain AI in simple terms"


def test_sidebar_controls_flow_into_request():
    with mock.patch("chat_api.requests.post", return_value=_ok("ok")) as p:
        at = make_app()
        at.run()
        at.sidebar.selectbox(key="selected_model").set_value("Gemma 4 31B 🆓 (Free)").run()
        at.sidebar.selectbox(key="persona").set_value("Senior software engineer").run()
        at.sidebar.slider(key="max_tokens").set_value(512).run()
        at.sidebar.slider(key="temperature").set_value(0.2).run()
        at.chat_input[0].set_value("q").run()
    assert not at.exception
    payload = p.call_args.kwargs["json"]
    assert payload["model"] == "google/gemma-4-31b-it:free"
    assert payload["max_tokens"] == 512 and payload["temperature"] == 0.2
    assert "senior software engineer" in payload["messages"][0]["content"]


def test_custom_system_prompt():
    with mock.patch("chat_api.requests.post", return_value=_ok("ok")) as p:
        at = make_app()
        at.run()
        at.sidebar.text_area(key="system_prompt").input("Be a pirate.").run()
        at.chat_input[0].set_value("q").run()
    assert p.call_args.kwargs["json"]["messages"][0]["content"] == "Be a pirate."


def test_regenerate_and_clear():
    with mock.patch("chat_api.requests.post", side_effect=[_ok("first"), _ok("second")]):
        at = make_app()
        at.run()
        at.chat_input[0].set_value("q").run()
        at.button(key="regen_btn").click().run()
    assert not at.exception
    assert [m["content"] for m in at.session_state.messages] == ["q", "second"]
    at.sidebar.button(key="clear_btn").click().run()
    assert at.session_state.messages == [] and at.session_state.total_tokens == 0


def test_history_cap_applied_in_request():
    with mock.patch("chat_api.requests.post", return_value=_ok("ok")) as p:
        at = make_app()
        at.run()
        at.session_state.messages = [{"role": "user", "content": str(i)} for i in range(30)]
        at.chat_input[0].set_value("last").run()
    sent = p.call_args.kwargs["json"]["messages"]
    assert len(sent) == 1 + chat_api.MAX_HISTORY_MESSAGES and sent[-1]["content"] == "last"


@pytest.mark.parametrize(
    "exc,needle",
    [
        (chat_api.requests.exceptions.Timeout, "timed out"),
        (chat_api.requests.exceptions.ConnectionError, "Cannot connect"),
    ],
)
def test_network_errors(exc, needle):
    with mock.patch("chat_api.requests.post", side_effect=exc):
        at = make_app()
        at.run()
        at.chat_input[0].set_value("q").run()
    assert not at.exception and at.session_state.messages == []
    assert any(needle in e.value for e in at.error)


def test_stream_error_chunk_and_empty_reply():
    with mock.patch("chat_api.requests.post", return_value=FakeResp(chunks=[{"error": {"message": "overloaded"}}])):
        at = make_app()
        at.run()
        at.chat_input[0].set_value("q").run()
    assert any("overloaded" in e.value for e in at.error)
    with mock.patch("chat_api.requests.post", return_value=FakeResp(chunks=[])):
        at = make_app()
        at.run()
        at.chat_input[0].set_value("q").run()
    assert any("empty reply" in e.value for e in at.error) and at.session_state.messages == []


@pytest.mark.parametrize("status,needle", [(402, "credits"), (429, "Rate limited")])
def test_other_http_errors(status, needle):
    with (
        mock.patch("chat_api.requests.post", return_value=FakeResp(status, body={"error": {"message": "x"}})),
        pytest.raises(chat_api.ChatError, match=needle),
    ):
        list(chat_api.stream_chat("k", [], "m", "s"))
