"""Runner module for executing the financial RAG pipeline."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from typing import Any

from codebase.agentmemory.cachememory import CacheMemory
from codebase.agentrunpipeline.answercomposer import FinancialAnswerComposer
from codebase.agentrunpipeline.citationdebugger import Citation, CitationDebugWriter, ToolTrace
from codebase.agentrunpipeline.contextbuilder import FinancialContextBuilder
from codebase.agentrunpipeline.models import AnswerGenerator, RAGResponse
from codebase.agentrunpipeline.querycheckpointer import QueryCheckpointer
from codebase.agentrunpipeline.queryplanner import FinancialQueryPlanner
from codebase.agentrunpipeline.retrievaltools import FinancialRetrievalTools

from config import CONFIG
from codebase.vectordb.chromastore import CHROMASTORE


class FinancialPipelineRunner:
    """End-to-end financial RAG runner over the existing Chroma vector store."""

    def __init__(
        self,
        chroma_store: Any | None = None,
        debug_output_dir: str | Path = "rag_debug",
        answer_generator: AnswerGenerator | None = None,
        cache_memory: CacheMemory | None = None,
        use_cache: bool = True,
        use_default_llm: bool = False,
    ) -> None:
        self.checkpointer = QueryCheckpointer()
        self.planner = FinancialQueryPlanner()
        self.retrieval_tools = FinancialRetrievalTools(chroma_store)
        self.context_builder = FinancialContextBuilder()
        self.answer_composer = FinancialAnswerComposer(answer_generator, use_default_llm=use_default_llm)
        self.debug_writer = CitationDebugWriter(debug_output_dir)
        self.cache_memory = cache_memory or CacheMemory()
        self.use_cache = use_cache

    def answer(
        self,
        question: str,
        company: str | None = None,
        year: int | str | None = None,
        doc_type: str | None = None,
        extra_filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> RAGResponse:
        """Run validation, cache lookup, retrieval, answer composition, and debug JSON writing."""
        
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
                cache={"enabled": self.use_cache, "hit": False, "lookup_skipped": "checkpointer_rejected"},
            )

        plan = self.planner.plan(
            question=question,
            company=company,
            year=year,
            doc_type=doc_type,
            extra_filters=extra_filters,
        )
        cache_key, cache_payload = self.cache_memory.build_cache_key(
            question=question,
            company=company,
            year=year,
            doc_type=doc_type,
            extra_filters=extra_filters,
            top_k=top_k,
        )

        cached_response = None
        if self.use_cache:
            cached_response = self.cache_memory.get_cached_response(cache_key)
        cache_trace = self._build_cache_trace(cache_key, cache_payload, cached_response, self.use_cache)
        tools_used.append(cache_trace)

        if cached_response:
            response = self._response_from_cache(cached_response["response"])
            citations = self._citations_from_response(response)
            trace = self.debug_writer.build_trace(
                question=question,
                status="answered_from_cache",
                answer=response.answer,
                checkpointer=check,
                filters=plan.filters,
                expanded_queries=plan.expanded_queries,
                tools_used=tools_used,
                citations=citations,
            )
            debug_path = self.debug_writer.write_trace(trace)
            response.status = "answered_from_cache"
            response.debug_json_path = str(debug_path)
            response.tools_used = [tool.__dict__ for tool in tools_used]
            response.checkpointer = check
            response.cache = {
                "enabled": self.use_cache,
                "hit": True,
                "cache_key": cache_key,
                "source_debug_json_path": cached_response.get("debug_json_path"),
                "hit_count": cached_response.get("hit_count"),
            }
            return response

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
        response = RAGResponse(
            status=trace.status,
            answer=answer_text,
            citations=[citation.__dict__ for citation in citations],
            debug_json_path=str(debug_path),
            tools_used=[tool.__dict__ for tool in tools_used],
            checkpointer=check,
            cache={"enabled": self.use_cache, "hit": False, "cache_key": cache_key},
        )
        if self.use_cache and response.status in {"answered", "no_context_found"}:
            self.cache_memory.set_cached_response(
                cache_key=cache_key,
                normalized_payload=cache_payload,
                original_question=question,
                response=response,
                debug_json_path=str(debug_path),
            )
        return response

    @staticmethod
    def _build_cache_trace(
        cache_key: str,
        cache_payload: dict[str, Any],
        cached_response: dict[str, Any] | None,
        enabled: bool,
    ) -> ToolTrace:
        """Create a debug trace for the cache lookup step."""
        return ToolTrace(
            tool_name="query_cache_lookup",
            input={"cache_key": cache_key, "payload": cache_payload, "enabled": enabled},
            output_summary={
                "cache_hit": bool(cached_response),
                "hit_count": cached_response.get("hit_count") if cached_response else 0,
                "cached_status": cached_response.get("status") if cached_response else None,
            },
        )

    @staticmethod
    def _response_from_cache(payload: dict[str, Any]) -> RAGResponse:
        """Rehydrate a cached response dictionary into a RAGResponse."""
        return RAGResponse(
            status=payload.get("status", "answered_from_cache"),
            answer=payload.get("answer", ""),
            citations=payload.get("citations", []),
            debug_json_path=payload.get("debug_json_path"),
            tools_used=payload.get("tools_used", []),
            checkpointer=payload.get("checkpointer", {}),
            cache=payload.get("cache", {}),
        )

    @staticmethod
    def _citations_from_response(response: RAGResponse) -> list[Citation]:
        """Convert cached citation dictionaries back to Citation dataclasses."""
        citations: list[Citation] = []
        for citation in response.citations:
            citations.append(
                Citation(
                    source_id=citation.get("source_id", "cached_source"),
                    parent_id=citation.get("parent_id"),
                    child_id=citation.get("child_id"),
                    page_number=citation.get("page_number"),
                    company=citation.get("company"),
                    report_year=citation.get("report_year"),
                    doc_type=citation.get("doc_type"),
                    page_intent=citation.get("page_intent"),
                    distance=citation.get("distance"),
                    snippet=citation.get("snippet", ""),
                    metadata=citation.get("metadata", {}),
                )
            )
        return citations

if __name__ == "__main__":
    runner = FinancialPipelineRunner(CHROMASTORE)
    response = runner.answer("What is Revenue of Kalyan Jewellers in 2024", "KALYANKJIL", 2025, "ANNUAL_REPORT")
    print(f"\n{'='*60}")
    print(f"STATUS : {response.status}")
    print(f"ANSWER : {response.answer}")
    print(f"{'='*60}")