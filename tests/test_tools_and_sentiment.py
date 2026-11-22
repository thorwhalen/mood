"""Offline tests for mood.tools (news search-and-save) and mood.sentiment (injected analyzers)."""

import json
import re
import sys
import types

import pytest

from mood import sentiment, tools
from mood.util import add_egress


def test_mk_filename_safe():
    assert tools.mk_filename_safe("AI & chips: 2026/Q1") == "AI___chips__2026_Q1"


def test_get_key_shape():
    key = tools.get_key("a b", prefix="p/", suffix=".json")
    assert re.fullmatch(r"p/\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}__a_b\.json", key)
    day = tools.get_key("q", day_folder=True)
    assert re.match(r"\d{4}-\d{2}-\d{2}[/\\]\d{4}", day)


def test_current_time_str_format():
    assert re.fullmatch(r"\d{4}", tools.current_time_str("%Y"))


def test_search_and_save_into_a_dict_store():
    store = {}
    key = tools._search_and_save_news(
        "oil", store=store, search_func=lambda q: [f"result for {q}"],
        query_to_key=lambda q: f"k_{q}",
    )
    assert key == "k_oil" and store == {"k_oil": ["result for oil"]}


def test_search_and_save_into_a_folder(tmp_path):
    key = tools._search_and_save_news(
        "oil", store=str(tmp_path / "news"), search_func=lambda q: {"n": 1},
        query_to_key=lambda q: "sub/oil.json",
    )
    assert json.loads((tmp_path / "news" / "sub" / "oil.json").read_text()) == {"n": 1}


def test_resolve_to_file_if_it_is_one(tmp_path):
    f = tmp_path / "queries.json"
    f.write_text("[]")
    assert tools._resolve_to_file_if_it_is_one(str(f)) == str(f)
    assert tools._resolve_to_file_if_it_is_one(str(tmp_path / "missing.json")) is False
    assert tools._resolve_to_file_if_it_is_one("just a query") is False
    assert tools._resolve_to_file_if_it_is_one(42) is False


def test_search_and_save_news_with_queries_from_a_file(tmp_path, monkeypatch, capsys):
    qfile = tmp_path / "queries.json"
    qfile.write_text(json.dumps(["oil", "gold"]))
    seen = []
    monkeypatch.setitem(tools.search_news_from, "fake", lambda q: seen.append(q) or [q])
    store = {}
    tools.search_and_save_news(str(qfile), source="fake", store=store, verbose=True)
    assert seen == ["oil", "gold"] and len(store) == 2
    assert all(k.startswith("fake") for k in store)
    assert "source='fake'" in capsys.readouterr().out
    tools.search_and_save_news("single", source="fake", store=store)
    assert seen[-1] == "single"
    with pytest.raises(AssertionError, match="Unknown news source"):
        tools.search_and_save_news("x", source="nope", store=store)


def test_sentiment_analysis_joins_headlines_and_uses_injected_analyzer():
    got = {}

    def analyze(headlines):
        got["h"] = headlines
        return {"a": 1}

    assert sentiment.sentiment_analysis(["a", "b"], analyze=analyze) == {"a": 1}
    assert got["h"] == "a\n\nb"
    sentiment.sentiment_analysis("solo", analyze=analyze)
    assert got["h"] == "solo"


def test_default_analyzer_is_built_on_aix_lazily(monkeypatch):
    calls = {}
    fake = types.ModuleType("aix")

    def prompt_func(prompt, **kw):
        calls.update(prompt=prompt, **kw)
        return lambda headlines: {"h": 5}

    fake.prompt_func = prompt_func
    monkeypatch.setitem(sys.modules, "aix", fake)
    assert sentiment.sentiment_analysis("h") == {"h": 5}
    assert calls["output_schema"] is dict and "{headlines}" in calls["prompt"]
    assert calls["name"] == "sentiment_analysis"


def test_flair_score_sign_follows_label(monkeypatch):
    class Label:
        def __init__(self, value, score):
            self.value, self.score = value, score

    class Sentence:
        def __init__(self, text):
            self.labels = [LABEL]

    class Classifier:
        def predict(self, s):
            pass

    flair = types.ModuleType("flair")
    data = types.ModuleType("flair.data")
    data.Sentence = Sentence
    monkeypatch.setitem(sys.modules, "flair", flair)
    monkeypatch.setitem(sys.modules, "flair.data", data)
    monkeypatch.setattr(sentiment, "_flair_text_classifier", lambda model="x": Classifier())
    global LABEL
    for value, expected in [("POSITIVE", 0.9), ("negative", -0.9), ("MIXED", 0.0)]:
        LABEL = Label(value, 0.9)
        assert sentiment.flair_sentiment_score("t") == expected


def test_add_egress():
    @add_egress(str)
    def f(x):
        return x + 1

    assert f(3) == "4" and f.__wrapped__(3) == 4
