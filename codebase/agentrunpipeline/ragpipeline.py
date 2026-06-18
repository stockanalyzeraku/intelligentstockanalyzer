"""Backward-compatible imports for the financial RAG pipeline.

Prefer importing :class:`FinancialPipelineRunner` from
``codebase.agentrunpipeline.financialpipelinerunner`` for new code.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from codebase.agentrunpipeline.answercomposer import FinancialAnswerComposer
from codebase.agentrunpipeline.contextbuilder import FinancialContextBuilder
from codebase.agentrunpipeline.financialpipelinerunner import FinancialPipelineRunner
from codebase.agentrunpipeline.models import AnswerGenerator, QueryPlan, RAGResponse
from codebase.agentrunpipeline.querycheckpointer import QueryCheckpointer
from codebase.agentrunpipeline.queryplanner import FinancialQueryPlanner
from codebase.agentrunpipeline.retrievaltools import FinancialRetrievalTools

FinancialRAGPipeline = FinancialPipelineRunner

__all__ = [
    "AnswerGenerator",
    "FinancialAnswerComposer",
    "FinancialContextBuilder",
    "FinancialPipelineRunner",
    "FinancialQueryPlanner",
    "FinancialRAGPipeline",
    "FinancialRetrievalTools",
    "QueryCheckpointer",
    "QueryPlan",
    "RAGResponse",
]
