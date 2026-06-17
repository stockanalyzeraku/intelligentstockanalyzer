from codebase.agentrunpipeline.querycheckpointer import QueryCheckpointer


def test_basic_metric_query_needs_more_information():
    result = QueryCheckpointer().validate("What is revenue")

    assert result["allowed"] is False
    assert result["reason"] == "query_too_basic"
    assert "company" in result["missing_context"]
    assert "year" in result["missing_context"]


def test_metric_only_query_needs_more_information():
    result = QueryCheckpointer().validate("PAT")

    assert result["allowed"] is False
    assert result["reason"] == "query_too_basic"


def test_specific_financial_query_is_accepted():
    result = QueryCheckpointer().validate(
        "What is revenue for KALYANKJIL in 2025?",
        company="KALYANKJIL",
        year=2025,
    )

    assert result["allowed"] is True
    assert result["reason"] == "accepted"
