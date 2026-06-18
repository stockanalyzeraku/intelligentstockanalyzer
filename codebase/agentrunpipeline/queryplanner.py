# """Financial query planning and synonym expansion."""

# from __future__ import annotations
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from typing import Any

# from codebase.agentrunpipeline.models import QueryPlan


# class FinancialQueryPlanner:
#     """Build metadata filters and financial query variants."""

#     METRIC_SYNONYMS: dict[str, list[str]] = {
#         "revenue": ["revenue", "total income", "income from operations", "sales", "topline"],
#         "profit": ["profit", "profit after tax", "PAT", "net profit", "profit for the year"],
#         "pat": ["PAT", "profit after tax", "net profit", "profit for the year"],
#         "ebitda": ["EBITDA", "operating profit", "earnings before interest tax depreciation amortisation"],
#         "debt": ["debt", "borrowings", "loans", "lease liabilities", "finance costs"],
#         "cash flow": ["cash flow", "cash generated from operations", "cash flow from operating activities"],
#         "risk": ["risk", "risk factors", "risk management", "threats", "uncertainties"],
#     }

#     def plan(
#         self,
#         question: str,
#         company: str | None = None,
#         year: int | str | None = None,
#         doc_type: str | None = None,
#         extra_filters: dict[str, Any] | None = None,
#     ) -> QueryPlan:
#         """Create a first-pass retrieval plan."""
#         filters: dict[str, Any] = {}
#         if company:
#             filters["company"] = company
#         if year:
#             filters["year"] = year
#         if doc_type:
#             filters["doc_type"] = doc_type
#         if extra_filters:
#             filters.update({k: v for k, v in extra_filters.items() if v is not None})

#         expanded_queries = self.expand_queries(question)
#         return QueryPlan(question=question, filters=filters, expanded_queries=expanded_queries)

#     def expand_queries(self, question: str) -> list[str]:
#         """Expand a question with financial synonyms while preserving the original."""
#         variants = [question]
#         lower_question = question.lower()
#         for metric, synonyms in self.METRIC_SYNONYMS.items():
#             if metric in lower_question:
#                 variants.extend(synonyms)
#         return list(dict.fromkeys(variants))
