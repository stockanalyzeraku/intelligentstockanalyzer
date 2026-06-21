"""Background RAG worker pipeline built with LangChain model components."""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from codebase.ragrun.cache_manager import RAGCacheManager
from codebase.ragrun.checkpointer import RuleBasedCheckpointer
from codebase.ragrun.config import RAGRUN_CONFIG
from codebase.ragrun.debugger import RAGDebugger
from codebase.ragrun.llm_router import LLMRouter
from codebase.ragrun.retriever import ChromaRAGRetriever
from codebase.ragrun.schemas import RAGWorkerResponse
from inputvalidator import InputValidator
from logger import get_logger
from healthcheck import assert_system_health

logger = get_logger(__name__)


class BackgroundRAGWorker:
    """Run the production RAG flow for one background query."""

    def __init__(
        self,
        checkpointer: RuleBasedCheckpointer | None = None,
        cache_manager: RAGCacheManager | None = None,
        retriever: ChromaRAGRetriever | None = None,
        llm_router: LLMRouter | None = None,
        debugger: RAGDebugger | None = None,
    ) -> None:
        self.checkpointer = checkpointer or RuleBasedCheckpointer()
        self.cache_manager = cache_manager or RAGCacheManager()
        self.retriever = retriever or ChromaRAGRetriever()
        self.llm_router = llm_router or LLMRouter()
        self.debugger = debugger or RAGDebugger()

    def answer(self, query: str, top_k: int | None = 5) -> RAGWorkerResponse:
        """Answer one user query and always write a JSON debug file."""
        assert_system_health(include_llm=True)
        query = InputValidator.validate_question(query)
        limit = InputValidator.validate_top_k(top_k, default=RAGRUN_CONFIG.top_k, max_value=RAGRUN_CONFIG.top_k)
        logger.process_event("rag_query_received", "rag", query_preview=query[:120], top_k=limit)
        
        with logger.timed("rag_answer", query_preview=query[:120], top_k=limit):
            check = self.checkpointer.validate(query)

            if not check.allowed:
                logger.process_event("rag_validation_rejected", "rag", status="skipped", reason=check.reason)
                return self._reject_query(query, check.to_dict())

            cache_key, cache_payload = self.cache_manager.build_key(query, top_k=limit)
            # cached = self.cache_manager.get(cache_key)

            # if cached:
            #     return self._answer_from_cache(query, check.to_dict(), cache_key, cached)

            chunks = self.retriever.search(query, top_k=limit)
            
            logger.process_event("retrieval_completed", "rag", chunks_found=len(chunks), top_k=limit)
            if not chunks:
                return self._answer_no_context(query, check.to_dict(), cache_key, cache_payload)
            print(type(chunks))
            for chunk in chunks:
                print(f"Chunk Information : {chunk}")
                print(f"Chunk Type : {type(chunk)}")

            model_result = self.llm_router.answer(query, chunks)
            response = RAGWorkerResponse(
                status="answered" if model_result.provider_used else "llm_failed",
                answer=model_result.answer,
                source="llm",
                debug_json_path=None,
                checkpointer=check.to_dict(),
                cache={"enabled": True, "hit": False, "cache_key": cache_key},
                model=model_result.to_dict(),
                retrieved_chunks=[chunk.to_dict() for chunk in chunks],
            )
            debug_path = self.debugger.write(
                query,
                {
                    "status": response.status,
                    "checkpointer": response.checkpointer,
                    "cache": response.cache,
                    "answer_source": "llm",
                    "model": response.model,
                    "documents_used": response.retrieved_chunks,
                    "answer": response.answer,
                },
            )
            response.debug_json_path = debug_path
            self.cache_manager.set(cache_key, cache_payload, query, response)
            logger.process_event("rag_answer_completed", "rag", status=response.status, provider=response.model.get("provider_used"), chunks_used=len(response.retrieved_chunks))
            return response

    def _reject_query(self, query: str, check: dict) -> RAGWorkerResponse:
        debug_path = self.debugger.write(
            query,
            {
                "status": "needs_more_information",
                "checkpointer": check,
                "cache": {"lookup_skipped": True},
                "answer_source": "checkpointer",
            },
        )
        return RAGWorkerResponse(
            status="needs_more_information",
            answer=check["message"],
            source="checkpointer",
            debug_json_path=debug_path,
            checkpointer=check,
            cache={"enabled": True, "hit": False, "lookup_skipped": True},
        )

    def _answer_from_cache(
        self,
        query: str,
        check: dict,
        cache_key: str,
        cached: dict,
    ) -> RAGWorkerResponse:
        cached_response = cached.get("response", {})
        debug_path = self.debugger.write(
            query,
            {
                "status": "answered_from_cache",
                "checkpointer": check,
                "cache": {
                    "hit": True,
                    "cache_key": cache_key,
                    "hit_count": cached.get("hit_count"),
                    "message": "Cache hit for answer.",
                },
                "answer_source": "cache",
                "cached_debug_json_path": cached.get("debug_json_path"),
            },
        )
        return RAGWorkerResponse(
            status="answered_from_cache",
            answer=cached_response.get("answer", ""),
            source="cache",
            debug_json_path=debug_path,
            checkpointer=check,
            cache={
                "enabled": True,
                "hit": True,
                "cache_key": cache_key,
                "hit_count": cached.get("hit_count"),
            },
            model=cached_response.get("model", {}),
            retrieved_chunks=cached_response.get("retrieved_chunks", []),
        )

    def _answer_no_context(
        self,
        query: str,
        check: dict,
        cache_key: str,
        cache_payload: dict,
    ) -> RAGWorkerResponse:
        answer = "I was not able to find anything that could answer the question."
        debug_path = self.debugger.write(
            query,
            {
                "status": "no_context_found",
                "checkpointer": check,
                "cache": {"hit": False, "cache_key": cache_key},
                "retrieval": {"chunks_found": 0},
                "answer_source": "retriever",
            },
        )
        response = RAGWorkerResponse(
            status="no_context_found",
            answer=answer,
            source="retriever",
            debug_json_path=debug_path,
            checkpointer=check,
            cache={"enabled": True, "hit": False, "cache_key": cache_key},
            retrieved_chunks=[],
        )
        self.cache_manager.set(cache_key, cache_payload, query, response)
        return response


if __name__ == "__main__":
    worker = BackgroundRAGWorker()
    example = worker.answer("What is revenue for Kalyan Jewellers for 2023?")
    # print(example.to_dict())