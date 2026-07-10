"""Tests for the Gemini-primary, Groq-fallback generation logic."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import gemini


def _fake_settings(groq_key: str) -> SimpleNamespace:
    return SimpleNamespace(groq_api_key=groq_key, groq_model="test-groq",
                           generation_model="test-gemini")


def test_uses_gemini_when_it_succeeds(monkeypatch):
    monkeypatch.setattr(gemini, "_generate_gemini", lambda p: "Gemini answer.")
    monkeypatch.setattr(gemini, "_generate_groq", lambda p: pytest.fail("Groq should not be called"))
    monkeypatch.setattr(gemini, "settings", _fake_settings("key"))

    assert gemini.generate_answer("q", ["ctx"]) == "Gemini answer."


def test_falls_back_to_groq_when_gemini_fails(monkeypatch):
    def gemini_fails(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(gemini, "_generate_gemini", gemini_fails)
    monkeypatch.setattr(gemini, "_generate_groq", lambda p: "Groq answer.")
    monkeypatch.setattr(gemini, "settings", _fake_settings("groq-key"))

    assert gemini.generate_answer("q", ["ctx"]) == "Groq answer."


def test_no_fallback_configured_raises(monkeypatch):
    def gemini_fails(prompt):
        raise RuntimeError("429")

    monkeypatch.setattr(gemini, "_generate_gemini", gemini_fails)
    monkeypatch.setattr(gemini, "settings", _fake_settings(""))  # no Groq key

    with pytest.raises(gemini.GeminiError):
        gemini.generate_answer("q", ["ctx"])


def test_raises_when_both_providers_fail(monkeypatch):
    def boom(prompt):
        raise RuntimeError("provider down")

    monkeypatch.setattr(gemini, "_generate_gemini", boom)
    monkeypatch.setattr(gemini, "_generate_groq", boom)
    monkeypatch.setattr(gemini, "settings", _fake_settings("groq-key"))

    with pytest.raises(gemini.GeminiError):
        gemini.generate_answer("q", ["ctx"])


def test_empty_gemini_response_triggers_fallback(monkeypatch):
    monkeypatch.setattr(gemini, "_generate_gemini", lambda p: "")
    monkeypatch.setattr(gemini, "_generate_groq", lambda p: "Groq answer.")
    monkeypatch.setattr(gemini, "settings", _fake_settings("groq-key"))

    assert gemini.generate_answer("q", ["ctx"]) == "Groq answer."


def test_groq_retries_after_transient_failure(monkeypatch):
    monkeypatch.setattr(gemini, "_generate_gemini", lambda p: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(gemini, "settings", _fake_settings("groq-key"))
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)  # no real delay in tests

    attempts = {"n": 0}

    def flaky_groq(prompt):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("temporary rate limit")
        return "Groq answer."

    monkeypatch.setattr(gemini, "_generate_groq", flaky_groq)

    assert gemini.generate_answer("q", ["ctx"]) == "Groq answer."
    assert attempts["n"] == 2  # first attempt failed, second succeeded


def test_groq_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(gemini, "_generate_gemini", lambda p: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(gemini, "settings", _fake_settings("groq-key"))
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)

    attempts = {"n": 0}

    def always_fails(prompt):
        attempts["n"] += 1
        raise RuntimeError("still down")

    monkeypatch.setattr(gemini, "_generate_groq", always_fails)

    with pytest.raises(gemini.GeminiError):
        gemini.generate_answer("q", ["ctx"])
    assert attempts["n"] == gemini._GROQ_MAX_ATTEMPTS
