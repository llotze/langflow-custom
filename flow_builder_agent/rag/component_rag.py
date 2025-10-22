"""
RAG (Retrieval-Augmented Generation) system for the Flow Builder Agent.
This system queries existing Langflow components instead of using hardcoded definitions.
"""

import json
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path

# Use existing Langflow component system
try:
    from langflow.interface.types import get_type_dict
    LANGFLOW_AVAILABLE = True
except ImportError:
    try:
        # Fallback to lfx if langflow not available
        from lfx.interface.components import import_langflow_components, aget_all_types_dict
        LANGFLOW_AVAILABLE = True
        get_type_dict = None
    except ImportError:
        LANGFLOW_AVAILABLE = False
        get_type_dict = None


class ComponentRAG:
    """RAG system that queries existing Langflow components instead of hardcoded definitions."""
    
    # Common component name variations/aliases
    COMPONENT_ALIASES = {
        "OpenAI": "OpenAIModel",
        "Anthropic": "AnthropicModel",
        "Google": "GoogleGenerativeAIModel",
        "Gemini": "GoogleGenerativeAIModel",
        "Amazon": "AmazonS3Component",
        "S3": "AmazonS3Component",
        "Pinecone": "PineconeVectorStore",
        "Chroma": "ChromaVectorStore",
        "Weaviate": "WeaviateVectorStore",
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG system with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.components_cache = None
        self.component_embeddings = None
        self.component_names = None
        self.logger = logging.getLogger(__name__)
          # Initialize from existing Langflow system
        self._load_langflow_components()
        self._build_embeddings()
    
    def _load_langflow_components(self):
        """Load components from existing Langflow system."""
        try:
            if LANGFLOW_AVAILABLE and get_type_dict is not None:
                # Use get_type_dict to get all available components
                self.logger.info("Loading components using get_type_dict()...")
                self.components_cache = get_type_dict()
                self.logger.info(f"✅ Loaded {len(self.components_cache)} component categories from Langflow")
                
                # Log what we got
                total_components = sum(len(comps) if isinstance(comps, dict) else 0 
                                     for comps in self.components_cache.values())
                self.logger.info(f"   Total components across all categories: {total_components}")
                
                # Log first few categories
                for cat in list(self.components_cache.keys())[:5]:
                    comps = self.components_cache[cat]
                    if isinstance(comps, dict):
                        self.logger.info(f"   Category '{cat}': {len(comps)} components")
            else:
                self.logger.warning("Langflow interface not available, using fallback components")
                self._load_fallback_components()
        except Exception as e:
            self.logger.error(f"Failed to load Langflow components: {e}")
            import traceback
            traceback.print_exc()
            self._load_fallback_components()
    
    async def _async_load_components(self):
        """Async helper to load components from Langflow."""
        # Try to get components from import_langflow_components first
        langflow_result = await import_langflow_components()
        return langflow_result.get("components", {})
    
    async def async_reload_components(self):
        """Async method to reload components from Langflow system."""
        try:
            if LANGFLOW_AVAILABLE:
                self.components_cache = await self._async_load_components()
                self._build_embeddings()
                self.logger.info(f"Reloaded {len(self.components_cache)} component categories from Langflow")
                return True
            else:
                self.logger.warning("Langflow interface not available")
                return False
        except Exception as e:
            self.logger.error(f"Failed to reload Langflow components: {e}")
            return False

    def _load_fallback_components(self):
        """Fallback component definitions for development/testing."""
        self.components_cache = {
            "inputs": {
                "ChatInput": {
                    "description": "Receives chat messages from users",
                    "category": "inputs",
                    "display_name": "Chat Input"
                }
            },
            "outputs": {
                "ChatOutput": {
                    "description": "Sends messages to users",
                    "category": "outputs", 
                    "display_name": "Chat Output"
                }
            },
            "models": {
                "OpenAI": {
                    "description": "OpenAI language models",
                    "category": "models",
                    "display_name": "OpenAI"
                }            }
        }
    
    def resolve_component_name(self, name: str) -> str:
        """
        Resolve a component name to its actual Langflow component name.
        Handles common aliases and variations.
        
        Args:
            name: Component name or alias
            
        Returns:
            Actual component name in Langflow, or original name if not found
        """
        # Check if it's an alias
        if name in self.COMPONENT_ALIASES:
            resolved = self.COMPONENT_ALIASES[name]
            self.logger.info(f"Resolved component alias '{name}' → '{resolved}'")
            return resolved
        
        # Check if the name exists in our components cache
        if self.components_cache:
            for category, components in self.components_cache.items():
                if isinstance(components, dict) and name in components:
                    return name  # Name is correct as-is
        
        # Return original name if not found
        return name
    
    def _build_embeddings(self):
        """Build and cache embeddings for components."""
        if not self.components_cache:
            self.logger.warning("No components cache available for embeddings")
            return
            
        component_texts = []
        component_names = []
        
        self.logger.info("Building embeddings from component cache...")
        
        for category, components in self.components_cache.items():
            if isinstance(components, dict):
                for comp_name, comp_info in components.items():
                    # Create searchable text from component info
                    text_parts = [comp_name, category]
                    
                    # Add information from component metadata
                    if isinstance(comp_info, dict):
                        # Get display name
                        display_name = comp_info.get("display_name", comp_name)
                        if display_name != comp_name:
                            text_parts.append(display_name)
                        
                        # Get description
                        desc = comp_info.get("description", "")
                        if desc:
                            text_parts.append(desc)
                        
                        # Get documentation
                        docs = comp_info.get("documentation", "")
                        if docs:
                            text_parts.append(docs[:200])  # Limit doc length
                        
                        # Get base classes (useful for finding similar components)
                        base_classes = comp_info.get("base_classes", [])
                        if base_classes:
                            text_parts.extend(base_classes)
                    
                    # Create final search text
                    text = " ".join(text_parts)
                    component_texts.append(text)
                    component_names.append((category, comp_name, comp_info))
        
        if component_texts:
            self.logger.info(f"Creating embeddings for {len(component_texts)} components...")
            self.component_embeddings = self.model.encode(component_texts)
            self.component_names = component_names
            self.logger.info(f"✅ Created embeddings for {len(component_texts)} components")
        else:
            self.logger.warning("No component texts to create embeddings from")
    
    def search_components(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Tuple[str, Any, float]]:
        """
        Search for components matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            threshold: Minimum similarity threshold
              Returns:
            List of (component_name, component_info, similarity_score) tuples
        """
        self.logger.info(f"🔍 RAG SEARCH: Query='{query}', top_k={top_k}, threshold={threshold}")
        
        if self.component_embeddings is None or len(self.component_embeddings) == 0 or not self.component_names:
            self.logger.warning("❌ RAG: No component embeddings available!")
            self.logger.warning(f"   Embeddings exist: {self.component_embeddings is not None}")
            self.logger.warning(f"   Component names exist: {self.component_names is not None}")
            self.logger.warning(f"   Components cache: {len(self.components_cache) if self.components_cache else 0} categories")
            return []
        
        self.logger.info(f"✅ RAG: Searching through {len(self.component_names)} components")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.component_embeddings)[0]
        
        # Filter by threshold and get top matches
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= threshold:
                category, name, comp_info = self.component_names[idx]
                results.append((name, comp_info, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        final_results = results[:top_k]
        
        self.logger.info(f"✅ RAG: Found {len(final_results)} components above threshold")
        for name, _, score in final_results[:3]:
            self.logger.info(f"   - {name}: {score:.3f}")
        
        return final_results
    
    def search_patterns(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for flow patterns matching the query.
        Since we removed hardcoded patterns, return common patterns based on query.
        """
        common_patterns = {
            "simple_chat": {
                "name": "Simple Chat Bot",
                "description": "Basic conversational AI using OpenAI",
                "components": ["ChatInput", "OpenAI", "ChatOutput"],
                "use_case": "General purpose chat application"
            },
            "rag_qa": {
                "name": "Document Q&A System", 
                "description": "RAG system for answering questions about uploaded documents",
                "components": ["ChatInput", "FileLoader", "TextSplitter", "Embeddings", "VectorStore", "Retriever", "OpenAI", "ChatOutput"],
                "use_case": "Query documents and get relevant answers"
            },
            "approval_workflow": {
                "name": "Human Approval Workflow",
                "description": "Workflow that requires human approval before proceeding",
                "components": ["ChatInput", "OpenAI", "ApprovalComponent", "EmailSender", "ChatOutput"], 
                "use_case": "Content moderation and approval processes"
            }
        }
        
        # Simple pattern matching based on keywords
        results = []
        query_lower = query.lower()
        
        for pattern_name, pattern_info in common_patterns.items():
            score = 0.0
            
            # Calculate relevance based on keywords
            if any(word in query_lower for word in ["chat", "bot", "conversation"]) and "chat" in pattern_name:
                score = 0.8
            elif any(word in query_lower for word in ["document", "file", "rag", "question"]) and "rag" in pattern_name:
                score = 0.9
            elif any(word in query_lower for word in ["approval", "human", "review"]) and "approval" in pattern_name:
                score = 0.85
            
            if score >= threshold:
                results.append((pattern_name, pattern_info, score))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def get_component_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific component by exact name match."""
        if not self.components_cache:
            return None
            
        for category, components in self.components_cache.items():
            if isinstance(components, dict) and name in components:
                comp_info = components[name]
                return {
                    "name": name,
                    "category": category,
                    "info": comp_info
                }
        return None
    
    def get_components_by_category(self, category: str) -> List[Tuple[str, Any]]:
        """Get all components in a specific category."""
        if not self.components_cache or category not in self.components_cache:
            return []
            
        components = self.components_cache[category]
        if isinstance(components, dict):
            return [(name, comp_info) for name, comp_info in components.items()]
        return []
    def analyze_user_request(self, user_request: str) -> Dict[str, Any]:
        """
        Analyze user request to extract requirements and suggest components.
        
        Args:
            user_request: User's natural language request
            
        Returns:
            Analysis containing suggested components, patterns, and requirements
        """
        self.logger.info(f"🔍 RAG ANALYZE: Starting analysis for '{user_request}'")
        request_text = user_request.lower()
        
        analysis = {
            "request": user_request,
            "detected_requirements": [],
            "suggested_components": [],
            "estimated_complexity": "medium"
        }
        
        # Detect common requirements
        if any(word in request_text for word in ["chat", "conversation", "talk", "ask"]):
            analysis["detected_requirements"].append("conversational_interface")
        
        if any(word in request_text for word in ["file", "document", "pdf", "upload"]):
            analysis["detected_requirements"].append("file_processing")
        
        if any(word in request_text for word in ["search", "find", "retrieve", "lookup"]):
            analysis["detected_requirements"].append("information_retrieval")
        
        if any(word in request_text for word in ["email", "send", "notify", "notification"]):
            analysis["detected_requirements"].append("email_notification")
        
        if any(word in request_text for word in ["approval", "review", "human", "check"]):
            analysis["detected_requirements"].append("human_approval")
        
        self.logger.info(f"✅ RAG ANALYZE: Detected requirements: {analysis['detected_requirements']}")
        
        # Estimate complexity
        num_requirements = len(analysis["detected_requirements"])
        if num_requirements <= 1:
            analysis["estimated_complexity"] = "simple"
        elif num_requirements <= 3:
            analysis["estimated_complexity"] = "medium"
        else:
            analysis["estimated_complexity"] = "advanced"
        
        # Get suggested components
        self.logger.info(f"🔍 RAG ANALYZE: Searching for components...")
        relevant_components = self.search_components(user_request, top_k=8)
        analysis["suggested_components"] = [
            {
                "name": name,
                "category": getattr(comp, "category", "unknown") if hasattr(comp, "category") else "unknown",
                "relevance_score": score,
                "reason": f"Relevant for: {user_request[:50]}..."
            }
            for name, comp, score in relevant_components
        ]
        
        self.logger.info(f"✅ RAG ANALYZE: Analysis complete with {len(analysis['suggested_components'])} components")
        
        return analysis
    
    def get_compatible_components(self, component_name: str) -> List[Tuple[str, Any, str]]:
        """
        Get components that are compatible with the given component.
        
        Args:
            component_name: Name of the component to find compatible ones for
            
        Returns:
            List of (name, component, relationship_type) tuples
        """
        compatible = []
        base_component = self.get_component_by_name(component_name)
        
        if not base_component:
            return compatible
        
        # Simple compatibility logic based on categories and names
        for category, components in self.components_cache.items():
            if isinstance(components, dict):
                for name, comp in components.items():
                    if name == component_name:
                        continue
                        
                    relationship = None
                    
                    # Check for input/output relationships
                    if "Input" in component_name and "Output" in name:
                        relationship = "can_connect_to"
                    elif "Output" in component_name and "Input" in name:
                        relationship = "receives_from"
                    elif category == "models" and base_component["category"] in ["prompts", "chains"]:
                        relationship = "uses_model"
                    elif category == "embeddings" and base_component["category"] == "vector_stores":
                        relationship = "uses_embeddings"
                    elif category == base_component["category"]:
                        relationship = "similar_category"
                    
                    if relationship:
                        compatible.append((name, comp, relationship))
        
        # Return top 10 most compatible
        return compatible[:10]
    
    def get_all_components(self) -> List[Dict[str, Any]]:
        """Get all available components as a list of dictionaries."""
        components = []
        
        if not self.components_cache:
            return components
            
        for category, comp_dict in self.components_cache.items():
            if isinstance(comp_dict, dict):
                for name, comp_info in comp_dict.items():
                    component_data = {
                        "name": name,
                        "category": category
                    }
                    
                    # Add additional info if available
                    if isinstance(comp_info, dict):
                        component_data.update({
                            "description": comp_info.get("description", ""),
                            "display_name": comp_info.get("display_name", name)
                        })
                    elif hasattr(comp_info, 'description'):
                        component_data.update({
                            "description": getattr(comp_info, 'description', ''),
                            "display_name": getattr(comp_info, 'display_name', name)
                        })
                    
                    components.append(component_data)
        
        return components


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = ComponentRAG()
    
    # Test component search
    results = rag.search_components("chat with users and get their input")
    print("Component Search Results:")
    for name, comp, score in results:
        print(f"  {name}: score {score:.3f}")
    
    # Test pattern search
    pattern_results = rag.search_patterns("answer questions about course syllabus with approval")
    print("\nPattern Search Results:")
    for name, pattern, score in pattern_results:
        print(f"  {name}: {pattern['description']} (score: {score:.3f})")
    
    # Test user request analysis
    user_request = "I want a bot to answer student questions about their due dates using only my syllabus file, and I need the draft response sent to my inbox for final approval before it is sent to the student."
    
    analysis = rag.analyze_user_request(user_request)
    print(f"\nUser Request Analysis:")
    print(f"Requirements: {analysis['detected_requirements']}")
    print(f"Complexity: {analysis['estimated_complexity']}")
    print(f"Top Components: {[comp['name'] for comp in analysis['suggested_components'][:3]]}")
