"""Shared mocking fixtures for the mocked eval suite (test_pipeline_mocked.py).

These stub out exactly the third-party packages NOT available in arbitrary
environments (langchain, langchain_mistralai, pydantic, chromadb-backed
ChromaStore) so the DETERMINISTIC stages (clarification, series_tools,
enrichment, derived_metrics, followup, the cache, and the orchestrator's
wiring) can be exercised for real, every run, without needing real network
access or API keys.

This does NOT test whether a real Mistral model actually follows the
system prompts in query_understanding.py / synthesis.py - that requires
eval_real_api.py, run separately and occasionally against a real key.

Call install_mocks() once at the top of a test module, BEFORE importing
anything from codebase.agent.* that transitively imports langchain/pydantic
(i.e. before importing codebase.agent.pipeline, .schemas, .tools, etc).
"""

from __future__ import annotations

import sys
import types


def install_mocks(synth_answer_text: str = "Synthesized answer.") -> dict:
    """Install all third-party stubs into sys.modules.

    Returns a dict of call-count trackers and the fake agent instances, so
    test code can assert on how many times each stage was actually invoked
    and inspect what prompt text the synthesis agent received.

    Returns
    -------
    dict with keys:
        "qu_call_count"      : dict, {"n": int} - increments on every Query
                                Understanding agent.invoke() call
        "synth_call_count"   : dict, {"n": int} - increments on every
                                Synthesis agent.invoke() call
        "synth_prompts_seen" : list[str] - every prompt text the fake
                                Synthesis agent received, in call order
        "qu_responses"       : dict[str, QueryUnderstanding] - mutate this
                                to control what the fake Query Understanding
                                agent returns for a given exact query string
                                (see set_qu_response below)
        "set_qu_response"    : callable(query: str, **qu_kwargs) - registers
                                a canned QueryUnderstanding for an exact
                                query string match
        "chroma_search_results" : list[dict] - mutate this to control what
                                the fake ChromaStore returns from
                                query_children_with_parent_context
    """
    # --- pydantic stub: minimal but real field-default + validation behavior ---
    fake_pydantic = types.ModuleType("pydantic")

    class _FakeBaseModel:
        def __init__(self, **kwargs):
            for name, default in self.__class__._field_defaults.items():
                setattr(self, name, kwargs.get(name, default))

        def __repr__(self):
            fields = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__class__._field_defaults)
            return f"{self.__class__.__name__}({fields})"

    class _FieldMarker:
        def __init__(self, default):
            self.default = default

    def _Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return _FieldMarker(default_factory())
        return _FieldMarker(default if default is not ... else None)

    class BaseModel(_FakeBaseModel):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            defaults = {}
            for klass in reversed(cls.__mro__):
                for name, value in vars(klass).items():
                    if isinstance(value, _FieldMarker):
                        defaults[name] = value.default
            for name in getattr(cls, "__annotations__", {}):
                defaults.setdefault(name, None)
            cls._field_defaults = defaults

    BaseModel._field_defaults = {}
    fake_pydantic.BaseModel = BaseModel
    fake_pydantic.Field = _Field
    sys.modules["pydantic"] = fake_pydantic

    # --- langchain.tools stub: @tool decorator -> object with .invoke({...}) ---
    fake_langchain = types.ModuleType("langchain")
    fake_langchain_tools = types.ModuleType("langchain.tools")

    class _FakeToolWrapper:
        def __init__(self, func):
            self.func = func

        def invoke(self, kwargs):
            return self.func(**kwargs)

    def fake_tool_decorator(*_args, **_kwargs):
        def _decorator(func):
            return _FakeToolWrapper(func)
        return _decorator

    fake_langchain_tools.tool = fake_tool_decorator
    sys.modules["langchain"] = fake_langchain
    sys.modules["langchain.tools"] = fake_langchain_tools

    # --- config stub ---
    fake_config = types.ModuleType("config")

    class _C:
        MISTRAL_API_KEY = "fake-key-for-mocked-tests"

    fake_config.CONFIG = _C()
    sys.modules["config"] = fake_config

    # --- langchain.agents.create_agent stub: routes to fake QU or Synth agent ---
    qu_call_count = {"n": 0}
    synth_call_count = {"n": 0}
    synth_prompts_seen: list[str] = []
    qu_responses: dict[str, dict] = {}

    def set_qu_response(query: str, **qu_kwargs) -> None:
        """Register what the fake Query Understanding agent should return
        for an EXACT query string match. qu_kwargs are passed straight
        through to QueryUnderstanding(**qu_kwargs).
        """
        qu_responses[query] = qu_kwargs

    class _FakeQUAgent:
        def invoke(self, payload):
            qu_call_count["n"] += 1
            # Imported lazily so this module doesn't require schemas.py's
            # transitive pydantic import at module-load time, only once
            # install_mocks() has already stubbed pydantic.
            from codebase.agent.schemas import QueryUnderstanding

            content = payload["messages"][0]["content"]
            if content in qu_responses:
                return {"structured_response": QueryUnderstanding(**qu_responses[content])}
            return {
                "structured_response": QueryUnderstanding(
                    ambiguity_reason=f"(mock) no canned QU response registered for: {content!r}"
                )
            }

    class _FakeSynthAgent:
        def invoke(self, payload):
            synth_call_count["n"] += 1
            prompt = payload["messages"][0]["content"]
            synth_prompts_seen.append(prompt)
            return {
                "messages": [
                    type("M", (), {"content_blocks": [
                        {"type": "reasoning", "reasoning": "(mock reasoning, should be stripped)"},
                        {"type": "text", "text": synth_answer_text},
                    ]})()
                ]
            }

    _fake_qu_agent = _FakeQUAgent()
    _fake_synth_agent = _FakeSynthAgent()

    fake_agents_mod = types.ModuleType("langchain.agents")

    def fake_create_agent(**kwargs):
        if kwargs.get("response_format") is not None:
            return _fake_qu_agent
        return _fake_synth_agent

    fake_agents_mod.create_agent = fake_create_agent
    sys.modules["langchain.agents"] = fake_agents_mod

    fake_mistral_mod = types.ModuleType("langchain_mistralai")

    class _FakeChatMistralAI:
        def __init__(self, **_kwargs):
            pass

    fake_mistral_mod.ChatMistralAI = _FakeChatMistralAI
    sys.modules["langchain_mistralai"] = fake_mistral_mod

    # --- chromastore stub ---
    chroma_search_results: list[dict] = []

    sys.modules["codebase.vectordb"] = types.ModuleType("codebase.vectordb")
    fake_chromastore_mod = types.ModuleType("codebase.vectordb.chromastore")

    class _FakeChromaStore:
        def query_children_with_parent_context(self, **_kwargs):
            return chroma_search_results

    fake_chromastore_mod.ChromaStore = _FakeChromaStore
    sys.modules["codebase.vectordb.chromastore"] = fake_chromastore_mod

    return {
        "qu_call_count": qu_call_count,
        "synth_call_count": synth_call_count,
        "synth_prompts_seen": synth_prompts_seen,
        "qu_responses": qu_responses,
        "set_qu_response": set_qu_response,
        "chroma_search_results": chroma_search_results,
    }


def install_test_cache_db(db_path: str) -> None:
    """Point the module-level CacheMemory used by codebase.agent.pipeline at
    a throwaway test DB, WITHOUT modifying codebase/agentmemory at all.

    Must be called AFTER install_mocks() and BEFORE importing
    codebase.agent.pipeline (since pipeline.py creates its module-level
    _cache instance at import time).
    """
    import codebase.agentmemory as agentmemory_module

    _OriginalCacheMemory = agentmemory_module.CacheMemory

    class _TestCacheMemory(_OriginalCacheMemory):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("db_path", db_path)
            super().__init__(*args, **kwargs)

    agentmemory_module.CacheMemory = _TestCacheMemory
