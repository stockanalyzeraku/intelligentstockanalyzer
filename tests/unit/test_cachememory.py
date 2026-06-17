from codebase.agentmemory.cachememory import CacheMemory
from codebase.agentrunpipeline.models import RAGResponse


def _response() -> RAGResponse:
    return RAGResponse(
        status="answered",
        answer="Revenue was Rs 100 crore.",
        citations=[{"source_id": "source_1", "snippet": "Revenue was Rs 100 crore."}],
        debug_json_path="rag_debug/sample.json",
        tools_used=[{"tool_name": "child_parent_search"}],
        checkpointer={"allowed": True},
        cache={"hit": False},
    )


def test_cache_key_is_stable_and_context_sensitive(tmp_path):
    cache = CacheMemory(db_path=tmp_path / "cache.db")

    key_1, payload_1 = cache.build_cache_key(" What  is Revenue? ", company="test", year=2025)
    key_2, payload_2 = cache.build_cache_key("what is revenue?", company="TEST", year="2025")
    key_3, _ = cache.build_cache_key("what is revenue?", company="OTHER", year="2025")

    assert key_1 == key_2
    assert payload_1 == payload_2
    assert key_1 != key_3


def test_set_and_get_cached_response_updates_hit_count(tmp_path):
    cache = CacheMemory(db_path=tmp_path / "cache.db")
    cache_key, payload = cache.build_cache_key("What is revenue for TEST in 2025?", company="TEST", year=2025)

    cache.set_cached_response(cache_key, payload, "What is revenue for TEST in 2025?", _response())
    first_hit = cache.get_cached_response(cache_key)
    second_hit = cache.get_cached_response(cache_key)

    assert first_hit is not None
    assert first_hit["response"]["answer"] == "Revenue was Rs 100 crore."
    assert first_hit["hit_count"] == 1
    assert second_hit["hit_count"] == 2


def test_expired_cache_entry_is_not_returned(tmp_path):
    cache = CacheMemory(db_path=tmp_path / "cache.db", default_ttl_seconds=-1)
    cache_key, payload = cache.build_cache_key("What is revenue for TEST in 2025?", company="TEST", year=2025)

    cache.set_cached_response(cache_key, payload, "What is revenue for TEST in 2025?", _response())

    assert cache.get_cached_response(cache_key) is None
