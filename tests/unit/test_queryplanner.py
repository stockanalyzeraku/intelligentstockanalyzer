from codebase.agentrunpipeline.queryplanner import FinancialQueryPlanner


def test_revenue_query_expands_financial_synonyms():
    plan = FinancialQueryPlanner().plan(
        "What is revenue for TEST in 2025?",
        company="TEST",
        year=2025,
        doc_type="ANNUAL_REPORT",
    )

    assert plan.filters == {"company": "TEST", "year": 2025, "doc_type": "ANNUAL_REPORT"}
    assert "What is revenue for TEST in 2025?" in plan.expanded_queries
    assert "total income" in plan.expanded_queries
    assert "income from operations" in plan.expanded_queries
    assert "sales" in plan.expanded_queries


def test_extra_filters_are_merged_without_none_values():
    plan = FinancialQueryPlanner().plan(
        "What are the key risks?",
        company="TEST",
        extra_filters={"page_intent": "risk", "unused": None},
    )

    assert plan.filters["company"] == "TEST"
    assert plan.filters["page_intent"] == "risk"
    assert "unused" not in plan.filters
    assert "risk factors" in plan.expanded_queries
