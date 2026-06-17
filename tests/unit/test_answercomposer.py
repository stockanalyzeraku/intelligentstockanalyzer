import pytest

from codebase.agentrunpipeline.answercomposer import FallbackAnswerGenerator, FinancialAnswerComposer
from codebase.agentrunpipeline.citationdebugger import Citation


class FakeProvider:
    def __init__(self, name, answer=None, error=None):
        self.name = name
        self.answer = answer
        self.error = error
        self.calls = 0

    def generate(self, question, context, records):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def _citation():
    return Citation(
        source_id="Source 1",
        parent_id="parent-1",
        child_id="child-1",
        page_number=10,
        company="TEST",
        report_year=2025,
        doc_type="annual_report",
        page_intent="financial_highlights",
        distance=0.1,
        snippet="Revenue was Rs 100 crore in FY2025.",
        metadata={},
    )


def test_fallback_answer_generator_uses_google_when_mistral_fails():
    mistral = FakeProvider("mistral", error=RuntimeError("mistral down"))
    google = FakeProvider("google", answer="Gemini answer [Source 1]")

    generator = FallbackAnswerGenerator([mistral, google])

    assert generator("question", "context", []) == "Gemini answer [Source 1]"
    assert mistral.calls == 1
    assert google.calls == 1


def test_fallback_answer_generator_raises_when_all_models_fail():
    generator = FallbackAnswerGenerator([
        FakeProvider("mistral", error=RuntimeError("mistral down")),
        FakeProvider("google", error=RuntimeError("google down")),
    ])

    with pytest.raises(RuntimeError, match="All answer providers failed"):
        generator("question", "context", [])


def test_composer_falls_back_to_extractive_answer_when_generator_fails():
    composer = FinancialAnswerComposer(
        answer_generator=lambda question, context, records: (_ for _ in ()).throw(RuntimeError("provider failed"))
    )

    answer = composer.compose("question", "context", [], [_citation()])

    assert "Revenue was Rs 100 crore" in answer
    assert "LLM answer generation failed" in answer
