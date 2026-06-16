"""Runner module for executing the financial RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codebase.agentrunpipeline.answercomposer import FinancialAnswerComposer
from codebase.agentrunpipeline.citationdebugger import CitationDebugWriter, ToolTrace
from codebase.agentrunpipeline.contextbuilder import FinancialContextBuilder
from codebase.agentrunpipeline.models import AnswerGenerator, RAGResponse
from codebase.agentrunpipeline.querycheckpointer import QueryCheckpointer
from codebase.agentrunpipeline.queryplanner import FinancialQueryPlanner
from codebase.agentrunpipeline.retrievaltools import FinancialRetrievalTools


class FinancialPipelineRunner:
    """End-to-end financial RAG runner over the existing Chroma vector store."""

    def __init__(
        self,
        chroma_store: Any | None = None,
        debug_output_dir: str | Path = "rag_debug",
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.checkpointer = QueryCheckpointer()
        self.planner = FinancialQueryPlanner()
        self.retrieval_tools = FinancialRetrievalTools(chroma_store)
        self.context_builder = FinancialContextBuilder()
        self.answer_composer = FinancialAnswerComposer(answer_generator)
        self.debug_writer = CitationDebugWriter(debug_output_dir)

    def answer(
        self,
        question: str,
        company: str | None = None,
        year: int | str | None = None,
        doc_type: str | None = None,
        extra_filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> RAGResponse:
        """Run validation, retrieval, answer composition, and debug JSON writing."""
        check = self.checkpointer.validate(question, company=company, year=year, doc_type=doc_type)
        tools_used: list[ToolTrace] = []

        if not check["allowed"]:
            trace = self.debug_writer.build_trace(
                question=question,
                status="needs_more_information",
                answer=check["message"],
                checkpointer=check,
                filters={},
                expanded_queries=[],
                tools_used=tools_used,
                citations=[],
            )
            debug_path = self.debug_writer.write_trace(trace)
            return RAGResponse(
                status="needs_more_information",
                answer=check["message"],
                citations=[],
                debug_json_path=str(debug_path),
                tools_used=[],
                checkpointer=check,
            )

        plan = self.planner.plan(
            question=question,
            company=company,
            year=year,
            doc_type=doc_type,
            extra_filters=extra_filters,
        )
        records, retrieval_trace = self.retrieval_tools.child_parent_search(
            queries=plan.expanded_queries,
            filters=plan.filters,
            top_k=top_k,
        )
        tools_used.append(retrieval_trace)

        context, citations = self.context_builder.build(records)
        answer_text = self.answer_composer.compose(question, context, records, citations)
        trace = self.debug_writer.build_trace(
            question=question,
            status="answered" if citations else "no_context_found",
            answer=answer_text,
            checkpointer=check,
            filters=plan.filters,
            expanded_queries=plan.expanded_queries,
            tools_used=tools_used,
            citations=citations,
        )
        debug_path = self.debug_writer.write_trace(trace)
        return RAGResponse(
            status=trace.status,
            answer=answer_text,
            citations=[citation.__dict__ for citation in citations],
            debug_json_path=str(debug_path),
            tools_used=[tool.__dict__ for tool in tools_used],
            checkpointer=check,
        )
