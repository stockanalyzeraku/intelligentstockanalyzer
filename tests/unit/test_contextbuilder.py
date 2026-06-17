from codebase.agentrunpipeline.contextbuilder import FinancialContextBuilder
from tests.helpers.fake_chroma_store import FakeChromaStore


def test_contextbuilder_creates_context_and_citations():
    records = [FakeChromaStore.default_record()]

    context, citations = FinancialContextBuilder().build(records)

    assert "source_1" in context
    assert "Revenue was Rs 100 crore" in context
    assert len(citations) == 1
    assert citations[0].source_id == "source_1"
    assert citations[0].parent_id == "page_10_parent"
    assert citations[0].child_id == "page_10_parent_child_0"
    assert citations[0].company == "TEST"
    assert citations[0].report_year == 2025
    assert citations[0].page_number == 10
