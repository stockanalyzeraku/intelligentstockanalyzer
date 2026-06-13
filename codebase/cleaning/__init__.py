"""
cleaning/
=========
Pipeline package for annual report ingestion and pre-processing.

Modules
-------
cleanresult        : Shared data-container dataclasses (CleanResult, TableType).
tableinfo          : Markdown-table extraction and row-level chunking.
textcleaner        : Line-level OCR noise removal and table classification.
pageintent         : Rule-based page-intent tagger.
embeddingprepared  : Chunk splitting and ChromaDB ingestion helpers.

All modules import Config from the root-level config.py via a sys.path
insertion so that this package can live in any sub-folder of the project.
"""
from codebase.cleaning.cleanresult import CleanResult, TableType
from codebase.cleaning.tableinfo import TableExtractor
from codebase.cleaning.textcleaner import TextCleaner
from codebase.cleaning.pageintent import PageIntentTagger, IntentResult
from codebase.cleaning.embeddingprepared import EmbeddingPrepared
from codebase.cleaning.pipelinerunner import PipelineRunner

__all__ = [
    "CleanResult",
    "TableType",
    "TableExtractor",
    "TextCleaner",
    "PageIntentTagger",
    "IntentResult",
    "EmbeddingPrepared",
    "PipelineRunner"
]