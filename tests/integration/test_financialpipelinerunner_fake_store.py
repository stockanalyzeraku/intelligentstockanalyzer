from pathlib import Path

from codebase.agentmemory.cachememory import CacheMemory
from codebase.agentrunpipeline.financialpipelinerunner import FinancialPipelineRunner
from codebase.agentrunpipeline.ragpipeline import FinancialRAGPipeline
from tests.helpers.fake_chroma_store import FakeChromaStore


def test_runner_blocks_basic_query_before_retrieval(tmp_path):
    store = FakeChromaStore()
    runner = FinancialPipelineRunner(
        chroma_store=store,
        debug_output_dir=tmp_path,
        cache_memory=CacheMemory(db_path=tmp_path / "cache.db"),
    )

    response = runner.answer("What is revenue")

    assert response.status == "needs_more_information"
    assert response.cache["lookup_skipped"] == "checkpointer_rejected"
    assert store.calls == 0
    assert Path(response.debug_json_path).exists()


def test_runner_cache_miss_then_cache_hit_skips_second_retrieval(tmp_path):
    store = FakeChromaStore()
    runner = FinancialPipelineRunner(
        chroma_store=store,
        debug_output_dir=tmp_path,
        cache_memory=CacheMemory(db_path=tmp_path / "cache.db"),
    )

    first = runner.answer("What is revenue for TEST in 2025?", company="TEST", year=2025)
    second = runner.answer("What is revenue for TEST in 2025?", company="TEST", year=2025)

    assert isinstance(runner, FinancialRAGPipeline)
    assert first.status == "answered"
    assert first.cache["hit"] is False
    assert second.status == "answered_from_cache"
    assert second.cache["hit"] is True
    assert store.calls == 1
    assert first.tools_used[0]["tool_name"] == "query_cache_lookup"
    assert first.tools_used[1]["tool_name"] == "child_parent_search"
    assert second.tools_used[0]["tool_name"] == "query_cache_lookup"
    assert all(tool["tool_name"] != "child_parent_search" for tool in second.tools_used)
    assert Path(first.debug_json_path).exists()
    assert Path(second.debug_json_path).exists()
