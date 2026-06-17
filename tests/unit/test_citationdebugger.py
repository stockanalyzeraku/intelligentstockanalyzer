import json

from codebase.agentrunpipeline.citationdebugger import Citation, CitationDebugWriter, ToolTrace


def test_citation_debug_writer_creates_readable_json(tmp_path):
    writer = CitationDebugWriter(tmp_path)
    trace = writer.build_trace(
        question="What is revenue for TEST in 2025?",
        status="answered",
        answer="Revenue was Rs 100 crore.",
        checkpointer={"allowed": True},
        filters={"company": "TEST", "year": 2025},
        expanded_queries=["revenue", "total income"],
        tools_used=[ToolTrace(tool_name="query_cache_lookup", output_summary={"cache_hit": False})],
        citations=[
            Citation(
                source_id="source_1",
                parent_id="page_10_parent",
                child_id="page_10_parent_child_0",
                page_number=10,
                company="TEST",
                report_year=2025,
                doc_type="ANNUAL_REPORT",
                page_intent="financial_highlights",
                distance=0.12,
                snippet="Revenue was Rs 100 crore.",
            )
        ],
    )

    path = writer.write_trace(trace)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.exists()
    assert payload["question"] == "What is revenue for TEST in 2025?"
    assert payload["tools_used"][0]["tool_name"] == "query_cache_lookup"
    assert payload["citations"][0]["source_id"] == "source_1"
