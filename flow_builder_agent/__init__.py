"""
Flow Builder Agent - Meta-AI for converting natural language to Langflow flows.

This package provides:
- Pydantic schemas for Langflow JSON structure
- RAG system for component retrieval  
- AI agent for flow generation
- Deployment utilities
"""

# Import only working components
from .simple_agent import SimpleFlowBuilderAgent
try:
    from .rag.component_rag import ComponentRAG
except ImportError:
    ComponentRAG = None

__version__ = "0.1.0"
__author__ = "Flow Builder Team"

__all__ = [
    "SimpleFlowBuilderAgent",
    "ComponentRAG",
]
