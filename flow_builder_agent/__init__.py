"""
Flow Builder Agent - Meta-AI for converting natural language to Langflow flows.

This package provides:
- RAG system for component retrieval
- Simple agent for flow generation
"""

from .simple_agent import SimpleFlowBuilderAgent

try:
    from .rag.component_rag import ComponentRAG
except Exception:  # pragma: no cover
    ComponentRAG = None

try:
    from .rag.gemini_rag import GeminiComponentRAG
except Exception:  # pragma: no cover
    GeminiComponentRAG = None

__version__ = "0.1.0"
__author__ = "Flow Builder Team"

__all__ = [
    "SimpleFlowBuilderAgent",
    "ComponentRAG",
    "GeminiComponentRAG",
]
