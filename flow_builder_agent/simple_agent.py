"""
Simplified Flow Builder Agent that generates basic Langflow JSON structures.
This version focuses on creating valid flow data that can be used with existing Langflow models.
"""

import json
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
# Optional Gemini dependency: import lazily
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

from .rag.component_rag import ComponentRAG
from .config import Config
# NEW: Provider-focused Gemini RAG grounding
try:
    from .rag.gemini_rag import GeminiComponentRAG
except Exception:  # pragma: no cover
    GeminiComponentRAG = None


class SimpleFlowBuilderAgent:
    """
    Simplified Flow Builder Agent that generates basic Langflow flow structures.
    
    This agent creates flow data dictionaries that can be used with existing
    Langflow FlowCreate models for database operations.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None, 
        provider: Optional[str] = None,
        langflow_api_url: Optional[str] = None,
        # Backwards compatibility
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the Simple Flow Builder Agent.
        
        Args:
            api_key: API key for the chosen provider. If None, uses config.
            model_name: Model to use. If None, uses config.
            provider: Either "gemini" or "openai". If None, uses config.
            langflow_api_url: URL of the Langflow API. If None, uses config.
            openai_api_key: Backwards compatibility - if provided, uses OpenAI
        """
        # Handle backwards compatibility
        if openai_api_key:
            provider = "openai"
            api_key = openai_api_key
        
        # Use config values as defaults
        self.provider = provider or Config.PROVIDER
        self.model_name = model_name or Config.MODEL_NAME
        self.langflow_api_url = langflow_api_url or Config.LANGFLOW_API_URL
        
        # Get API key
        if api_key is None:
            api_key = Config.get_api_key()
        
        self.rag = ComponentRAG(langflow_api_url=self.langflow_api_url)
        self.logger = logging.getLogger(__name__)
        # Initialize grounded Gemini RAG (optional) to reduce hallucinations
        self.grounded_rag = None
        if (provider or self.provider) == "gemini" and GeminiComponentRAG is not None:
            try:
                self.grounded_rag = GeminiComponentRAG(langflow_api_url=self.langflow_api_url)
                self.logger.info("🔎 GeminiComponentRAG initialized for provider-focused grounding")
            except Exception as e:
                self.logger.warning(f"GeminiComponentRAG not available: {e}")
        
        # Adjust model name if provider was changed via openai_api_key
        if openai_api_key and self.model_name.startswith("gemini"):
            self.model_name = "gpt-4o"
        
        # Initialize the appropriate client
        if self.provider == "gemini":
            if genai is None:
                raise ValueError(
                    "google-generativeai is not installed but provider='gemini'. Install it or switch provider."
                )
            if not api_key:
                raise ValueError("Gemini API key required. Set GOOGLE_API_KEY in .env file.")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.async_client = None
        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env file.")
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'")
        
        self.logger.info(f"✅ Initialized SimpleFlowBuilderAgent with {self.provider} ({self.model_name})")
    
    async def build_flow_async(self, user_request: str, flow_name: str = None) -> Dict[str, Any]:
        """
        Build a Langflow flow from a user request.
        
        Args:
            user_request: Natural language description of desired workflow
            flow_name: Optional name for the flow
            
        Returns:
            Dictionary with flow data suitable for FlowCreate
        """
        try:
            self.logger.info(f"🚀 CHECKPOINT 1: Starting build_flow_async for: '{user_request}'")
            
            # Analyze request using RAG
            self.logger.info("🔍 CHECKPOINT 2: Analyzing request with RAG...")
            analysis = self.rag.analyze_user_request(user_request)
            self.logger.info(f"✅ CHECKPOINT 2 SUCCESS: Analysis complete - {analysis.get('detected_requirements', [])}")
            
            # Get relevant components
            self.logger.info("🔍 CHECKPOINT 3: Searching for relevant components...")
            relevant_components = self.rag.search_components(user_request, top_k=10)
            self.logger.info(f"✅ CHECKPOINT 3 SUCCESS: Found {len(relevant_components)} relevant components")
            if relevant_components:
                comp_names = [comp[0] for comp in relevant_components[:3]]
                self.logger.info(f"   Top components: {comp_names}")
            
            # Generate flow structure using LLM
            self.logger.info("🤖 CHECKPOINT 4: Generating flow with LLM...")
            flow_data = await self._generate_flow_data_async(
                user_request, analysis, relevant_components
            )
            self.logger.info(f"✅ CHECKPOINT 4 SUCCESS: Flow generated with {len(flow_data.get('nodes', []))} nodes")
            
            # Create the complete flow object for database
            result = {
                "name": flow_name or self._generate_flow_name(user_request),
                "description": self._generate_flow_description(user_request, analysis),
                "data": flow_data,
                "is_component": False,
            }
            self.logger.info(f"🎉 CHECKPOINT 5 FINAL: Flow build complete! Name: '{result['name']}'")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ ERROR in build_flow_async: {str(e)}")
            self.logger.exception("Full traceback:")
            # Return a basic fallback flow
            return self._create_fallback_flow(user_request, flow_name)
    
    def build_flow(self, user_request: str, flow_name: str = None) -> Dict[str, Any]:
        """Synchronous version of build_flow_async."""
        return asyncio.run(self.build_flow_async(user_request, flow_name))
    
    async def _generate_flow_data_async(
        self, 
        user_request: str, 
        analysis: Dict[str, Any], 
        relevant_components: List[Tuple[str, Any, float]]
    ) -> Dict[str, Any]:
        """
        Generate the flow data structure using OpenAI.
        
        Returns a dictionary with nodes, edges, and viewport that matches
        Langflow's expected flow.data structure.
        """
        # Prepare component information for the LLM
        component_info = self._format_components_for_llm(relevant_components)
        
        messages = [
            {"role": "system", "content": """You are a Langflow expert. Generate ONLY a JSON structure for flow data.

The JSON should have this structure:
{
  "nodes": [
    {
      "id": "ComponentType-RandomID",
      "type": "genericNode",
      "position": {"x": 100, "y": 100},
      "data": {
        "type": "ComponentType",
        "id": "ComponentType-RandomID",
        "display_name": "Component Display Name",
        "description": "Component description"
      }
    }
  ],
  "edges": [
    {
      "id": "reactflow__edge-source-target",
      "source": "source-node-id", 
      "target": "target-node-id",
      "sourceHandle": "output_name",
      "targetHandle": "input_name"
    }
  ],
  "viewport": {"x": 0, "y": 0, "zoom": 1}
}

Common component types to use:
- ChatInput: For user input (position x: 100)
- OpenAI: For language model (position x: 400) 
- ChatOutput: For displaying results (position x: 700)
- Prompt: For system prompts (position x: 100)
- File: For file handling (position x: 100)

Create a simple, working flow with proper connections between components.
Position nodes left to right: inputs → processing → outputs."""},
            {"role": "user", "content": f"""
Create a Langflow flow for: {user_request}

Available components: {component_info}
Requirements detected: {', '.join(analysis.get('detected_requirements', []))}

Return ONLY the JSON structure for the flow data, no other text.
            """}
        ]
        
        try:
            self.logger.info(f"🤖 CHECKPOINT 4a: Using provider: {self.provider}")
            
            if self.provider == "gemini":
                # Use Gemini API with grounding context (if available)
                self.logger.info("🤖 CHECKPOINT 4b: Calling Gemini API...")
                grounded_context = self._ground_with_gemini_rag(user_request)
                prompt = f"""You are a Langflow expert. Generate ONLY a JSON structure for flow data.

The JSON should have this structure:
{{
  "nodes": [...],
  "edges": [...],
  "viewport": {{"x": 0, "y": 0, "zoom": 1}}
}}

Each node must have:
- id: unique identifier (e.g., "node_1", "node_2")
- type: ALWAYS "genericNode"
- position: {{x: number, y: number}}
- data: {{
    "type": "EXACT_COMPONENT_NAME",  // USE THE EXACT COMPONENT NAME FROM THE ALLOWED LIST BELOW
    "id": "same_as_node_id",
    "node": {{}}
  }}

ALLOWED COMPONENTS (JSON lines):
{grounded_context or component_info}

CRITICAL: Use ONLY the listed components; if insufficient, return a minimal flow with ChatInput → OpenAIModel → ChatOutput.
Create a Langflow flow for: {user_request}

Space nodes horizontally: Input x=100-200, Processing x=400-600, Output x=800-900
Return ONLY valid JSON, no markdown, no explanation."""
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                content = response.text
                self.logger.info(f"✅ CHECKPOINT 4c: Gemini responded with {len(content)} characters")
                # Persist raw output for debugging
                kb_dir = (Path(__file__).resolve().parents[1] / "knowledge_base")
                kb_dir.mkdir(parents=True, exist_ok=True)
                (kb_dir / "last_llm_flow.json").write_text(content or "", encoding="utf-8")
            else:
                # Use OpenAI API
                self.logger.info("🤖 CHECKPOINT 4b: Calling OpenAI API...")
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )
                content = response.choices[0].message.content.strip()
                self.logger.info(f"✅ CHECKPOINT 4c: OpenAI responded with {len(content)} characters")
            
            # Clean up the response to extract JSON
            self.logger.info("🔍 CHECKPOINT 4d: Parsing JSON response...")
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            flow_data = json.loads(content)
            self.logger.info(f"✅ CHECKPOINT 4e: JSON parsed successfully")
            # Validate and clean the flow data
            flow_data = self._validate_flow_data(flow_data)
            # Hydrate nodes with real component templates to avoid 'No template' issues
            self._hydrate_flow_nodes(flow_data)
            # Enforce model component to match provider/user intent (e.g., Gemini → GoogleGenerativeAIModel)
            self._enforce_provider_model_choice(user_request, flow_data)
            # Ensure essential IO nodes are present
            self._ensure_io_nodes(flow_data)
            # Normalize edge handles to JSON-escaped strings expected by frontend
            self._normalize_edge_handles(flow_data)
            # Mirror parsed handles into edge.data for safety with older helpers
            try:
                for e in flow_data.get("edges", []):
                    if isinstance(e.get("sourceHandle"), str) and isinstance(e.get("targetHandle"), str):
                        e.setdefault("data", {})
                        # Keep the raw strings; frontend utilities will re-escape as needed
                        e["data"]["sourceHandle"] = e["sourceHandle"]
                        e["data"]["targetHandle"] = e["targetHandle"]
            except Exception:
                pass
            return flow_data
            
        except Exception as e:
            self.logger.error(f"Error generating flow with LLM: {e}")
            return self._create_basic_fallback_flow_data(user_request)
    
    def _format_components_for_llm(self, relevant_components: List[Tuple[str, Any, float]]) -> str:
        """Format component information for the LLM."""
        if not relevant_components:
            return "No specific components found, use basic components: ChatInput, OpenAI, ChatOutput"
        
        component_text = "Available Langflow Components (use these exact names in 'type' field):\n\n"
        for comp_name, component, score in relevant_components[:8]:  # Top 8 components
            # Extract component information
            if isinstance(component, dict):
                display_name = component.get('display_name', comp_name)
                description = component.get('description', 'No description')
                base_classes = component.get('base_classes', [])
                
                component_text += f"Component: {comp_name}\n"
                if display_name != comp_name:
                    component_text += f"  Display Name: {display_name}\n"
                component_text += f"  Description: {description}\n"
                if base_classes:
                    component_text += f"  Type: {', '.join(base_classes[:2])}\n"
                component_text += f"  Relevance: {score:.2f}\n\n"
            else:
                component_text += f"- {comp_name}: {getattr(component, 'description', 'No description')}\n"
        
        component_text += "\nIMPORTANT: Use the exact 'Component' name (e.g., 'ChatInput', 'OpenAIModel') in the 'type' field of each node.\n"
        return component_text
    
    def _validate_flow_data(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the flow data structure."""
        # Ensure required keys exist
        if "nodes" not in flow_data:
            flow_data["nodes"] = []
        if "edges" not in flow_data:
            flow_data["edges"] = []
        if "viewport" not in flow_data:
            flow_data["viewport"] = {"x": 0, "y": 0, "zoom": 1}
          # Validate nodes
        for node in flow_data["nodes"]:
            if "id" not in node:
                node["id"] = f"Node-{uuid.uuid4().hex[:8]}"
            if "type" not in node:
                node["type"] = "genericNode"
            if "position" not in node:
                node["position"] = {"x": 100, "y": 100}
            if "data" not in node:
                node["data"] = {
                    "type": "ChatInput",
                    "id": node["id"],
                    "display_name": "Chat Input",
                    "description": "Basic input component"
                }
            
            # Resolve component name aliases
            if "data" in node and "type" in node["data"]:
                original_type = node["data"]["type"]
                resolved_type = self.rag.resolve_component_name(original_type)
                if resolved_type != original_type:
                    node["data"]["type"] = resolved_type
                    self.logger.info(f"Resolved component '{original_type}' → '{resolved_type}' in node {node['id']}")
        
        # Validate edges
        node_ids = {node["id"] for node in flow_data["nodes"]}
        valid_edges = []
        
        for edge in flow_data["edges"]:
            if (edge.get("source") in node_ids and 
                edge.get("target") in node_ids):
                if "id" not in edge:
                    edge["id"] = f"reactflow__edge-{edge['source']}-{edge['target']}"
                valid_edges.append(edge)
        
        flow_data["edges"] = valid_edges
        
        return flow_data

    def _hydrate_flow_nodes(self, flow_data: Dict[str, Any]) -> None:
        """Populate node.data.node with real component definitions from the catalog.
        This fixes missing template errors in the UI and improves downstream validation.
        """
        if not isinstance(flow_data, dict) or not flow_data.get("nodes"):
            return
        # Ensure RAG has components loaded
        try:
            if not self.rag.components_cache:
                self.rag._load_langflow_components()
        except Exception as e:
            self.logger.warning(f"Could not ensure components cache: {e}")

        for node in flow_data.get("nodes", []):
            try:
                data = node.get("data", {})
                comp_type = data.get("type")
                if not comp_type:
                    continue
                # Resolve aliases (e.g., Amazon -> AmazonBedrockConverseModel)
                resolved_type = self.rag.resolve_component_name(comp_type)
                if resolved_type != comp_type:
                    data["type"] = resolved_type
                    self.logger.info(f"Hydrate: resolved '{comp_type}' → '{resolved_type}'")
                comp = self.rag.get_component_by_name(resolved_type)
                if comp and isinstance(comp.get("info"), dict):
                    # Attach the full component definition under data.node
                    data["node"] = comp["info"]
                    # If display_name missing in node, use component's
                    if "display_name" not in data and comp["info"].get("display_name"):
                        data["display_name"] = comp["info"]["display_name"]
                else:
                    # Leave an empty node dict to be safe
                    data.setdefault("node", {})
            except Exception as e:
                self.logger.warning(f"Hydrate failed for node {node.get('id')}: {e}")

    def _enforce_provider_model_choice(self, user_request: str, flow_data: Dict[str, Any]) -> None:
        """Adjust model nodes to reflect requested provider or current provider.
        - If provider is gemini or user mentions 'gemini'/'google', prefer GoogleGenerativeAIModel.
        - If user mentions 'amazon'/'bedrock', prefer AmazonBedrockConverseModel.
        - If user mentions 'openai', prefer OpenAIModel.
        """
        try:
            text = (user_request or "").lower()
            want_openai = ("openai" in text)
            want_gemini = (self.provider == "gemini") or ("gemini" in text or "google" in text)
            want_bedrock = ("amazon" in text or "bedrock" in text)

            model_nodes = [n for n in flow_data.get("nodes", []) if isinstance(n.get("data"), dict) and isinstance(n["data"].get("type"), str)]
            if not model_nodes:
                return

            def _switch_node_type(node: Dict[str, Any], new_type: str) -> bool:
                try:
                    node["data"]["type"] = new_type
                    comp = self.rag.get_component_by_name(new_type)
                    if comp and isinstance(comp.get("info"), dict):
                        node["data"]["node"] = comp["info"]
                        if "display_name" not in node["data"] and comp["info"].get("display_name"):
                            node["data"]["display_name"] = comp["info"]["display_name"]
                        return True
                except Exception as e:
                    self.logger.warning(f"Provider enforcement failed for node {node.get('id')}: {e}")
                return False

            changed = False
            # Highest priority: explicit OpenAI request
            if want_openai:
                for node in model_nodes:
                    t = node["data"].get("type", "")
                    if _switch_node_type(node, "OpenAIModel"):
                        self.logger.info(f"🔁 Replaced model '{t}' → 'OpenAIModel' due to user intent")
                        changed = True
                        break
            # Next: Amazon Bedrock
            if not changed and want_bedrock:
                for node in model_nodes:
                    t = node["data"].get("type", "")
                    if _switch_node_type(node, "AmazonBedrockConverseModel"):
                        self.logger.info(f"🔁 Replaced model '{t}' → 'AmazonBedrockConverseModel' due to user intent")
                        changed = True
                        break
            # Finally: Gemini based on provider or mention
            if not changed and want_gemini:
                for node in model_nodes:
                    t = node["data"].get("type", "")
                    if _switch_node_type(node, "GoogleGenerativeAIModel"):
                        self.logger.info(f"🔁 Replaced model '{t}' → 'GoogleGenerativeAIModel' to match provider")
                        changed = True
                        break
        except Exception as e:
            self.logger.warning(f"Model provider enforcement skipped due to error: {e}")

    def _ensure_io_nodes(self, flow_data: Dict[str, Any]) -> None:
        """Ensure ChatInput and ChatOutput nodes exist; add them if missing and hydrate from catalog."""
        try:
            nodes = flow_data.setdefault("nodes", [])
            # Presence checks
            has_input = any(isinstance(n.get("data"), dict) and n["data"].get("type") == "ChatInput" for n in nodes)
            has_output = any(isinstance(n.get("data"), dict) and n["data"].get("type") == "ChatOutput" for n in nodes)

            # Helper to create hydrated node by type
            def _make_node(node_id: str, comp_type: str, x: int, y: int) -> Dict[str, Any]:
                comp = self.rag.get_component_by_name(comp_type)
                info = comp.get("info") if comp else {}
                return {
                    "id": node_id,
                    "type": "genericNode",
                    "position": {"x": x, "y": y},
                    "data": {
                        "type": comp_type,
                        "id": node_id,
                        "node": info if isinstance(info, dict) else {}
                    }
                }

            # Add ChatInput if missing
            if not has_input:
                new_id = "ChatInput-1"
                nodes.insert(0, _make_node(new_id, "ChatInput", 100, 100))
                self.logger.info("➕ Inserted missing ChatInput node")
            # Add ChatOutput if missing
            if not has_output:
                new_id = "ChatOutput-1"
                nodes.append(_make_node(new_id, "ChatOutput", 850, 100))
                self.logger.info("➕ Inserted missing ChatOutput node")

            # Basic edge wiring if edges are empty or incomplete
            edges = flow_data.setdefault("edges", [])
            if not edges and len(nodes) >= 2:
                # Connect first → middle → last in sequence
                ordered_ids = [n.get("id") for n in nodes]
                for a, b in zip(ordered_ids, ordered_ids[1:]):
                    if a and b:
                        edges.append({
                            "id": f"reactflow__edge-{a}-{b}",
                            "source": a,
                            "target": b
                        })
                self.logger.info(f"🔗 Auto-wired {len(edges)} edges between sequential nodes")
        except Exception as e:
            self.logger.warning(f"IO node ensuring skipped due to error: {e}")

    def _normalize_edge_handles(self, flow_data: Dict[str, Any]) -> None:
        """Ensure edge.sourceHandle and edge.targetHandle are valid JSON strings.
        If a handle is missing or is a plain label (e.g., 'data_inputs'), construct
        proper handle objects from node templates and stringify them.
        """
        try:
            nodes = {n.get("id"): n for n in flow_data.get("nodes", [])}
            edges = flow_data.get("edges", [])

            def _is_json_like(s: Optional[str]) -> bool:
                return isinstance(s, str) and s.strip().startswith("{") and s.strip().endswith("}")

            for edge in edges:
                src_id = edge.get("source")
                tgt_id = edge.get("target")
                src = nodes.get(src_id)
                tgt = nodes.get(tgt_id)
                # Build targetHandle if missing or not JSON
                if tgt and not _is_json_like(edge.get("targetHandle")):
                    tgt_data = tgt.get("data", {})
                    tgt_node = tgt_data.get("node", {})
                    tmpl = tgt_node.get("template", {}) if isinstance(tgt_node, dict) else {}
                    # Try to guess field
                    field = None
                    if isinstance(edge.get("targetHandle"), str) and edge["targetHandle"]:
                        field = edge["targetHandle"]
                    # Prefer a field that accepts Message if absent
                    if not field and isinstance(tmpl, dict):
                        # Prefer required Message inputs, else first input
                        candidates = [k for k, v in tmpl.items() if isinstance(v, dict) and isinstance(v.get("input_types"), list)]
                        # order by whether it accepts Message
                        candidates.sort(key=lambda k: ("Message" not in (tmpl[k].get("input_types") or []), not tmpl[k].get("required", False)))
                        field = candidates[0] if candidates else None
                    if field and field in tmpl:
                        tv = tmpl[field]
                        input_types = tv.get("input_types") or []
                        handle_obj = {
                            "type": (input_types[0] if input_types else tv.get("type")),
                            "fieldName": field,
                            "id": tgt_data.get("id") or tgt.get("id"),
                            "inputTypes": input_types,
                        }
                        # JSON stringify (quotes are fine; frontend escapes internally)
                        edge["targetHandle"] = json.dumps(handle_obj, ensure_ascii=False)
                        self.logger.info(f"Normalized targetHandle for edge {edge.get('id')} → field '{field}'")
                # Build sourceHandle if missing or not JSON
                if src and not _is_json_like(edge.get("sourceHandle")):
                    src_data = src.get("data", {})
                    src_node = src_data.get("node", {})
                    outputs = src_node.get("outputs") or []
                    out = outputs[0] if outputs else None
                    if isinstance(out, dict):
                        handle_obj = {
                            "id": src_data.get("id") or src.get("id"),
                            "dataType": src_data.get("type"),
                            "name": out.get("name", "text_output"),
                            "output_types": out.get("types", []),
                        }
                        edge["sourceHandle"] = json.dumps(handle_obj, ensure_ascii=False)
                        self.logger.info(f"Normalized sourceHandle for edge {edge.get('id')} → output '{handle_obj['name']}'")
        except Exception as e:
            self.logger.warning(f"Edge handle normalization skipped due to error: {e}")

    def _ground_with_gemini_rag(self, user_request: str) -> str:
        """Build a compact grounding context using provider-focused components.
        Returns JSON-lines text of allowed components.
        """
        if not self.grounded_rag:
            return ""
        try:
            self.grounded_rag.load_catalog()
            self.grounded_rag.build_index()
            matches = self.grounded_rag.retrieve(user_request, top_k=12, threshold=0.2)
            items = [m[0] for m in matches]
            context_lines = []
            for d in items:
                context_lines.append(json.dumps({
                    "category": d.get("category"),
                    "name": d.get("name"),
                    "display_name": d.get("display_name"),
                    "description": d.get("description"),
                    "base_classes": d.get("base_classes", []),
                    "inputs": d.get("inputs", [])
                }, ensure_ascii=False))
            grounded = "\n".join(context_lines)
            # Persist for debugging
            kb_dir = (Path(__file__).resolve().parents[1] / "knowledge_base")
            kb_dir.mkdir(parents=True, exist_ok=True)
            (kb_dir / "grounded_context.jsonl").write_text(grounded, encoding="utf-8")
            self.logger.info("📝 Saved grounded context to knowledge_base/grounded_context.jsonl")
            return grounded
        except Exception as e:
            self.logger.warning(f"Grounding step failed: {e}")
            return ""
    
    def _generate_flow_name(self, user_request: str) -> str:
        """Generate a name for the flow based on the user request."""
        # Extract key words and create a title
        words = user_request.split()[:4]  # First 4 words
        return " ".join(word.capitalize() for word in words if word.isalnum())
    
    def _generate_flow_description(self, user_request: str, analysis: Dict[str, Any]) -> str:
        """Generate a description for the flow."""
        requirements = analysis.get('detected_requirements', [])
        if requirements:
            req_text = ", ".join(req.replace('_', ' ') for req in requirements)
            return f"Flow for {user_request}. Handles: {req_text}."
        return f"Flow for {user_request}"
    
    def _create_fallback_flow(self, user_request: str, flow_name: str = None) -> Dict[str, Any]:
        """Create a basic fallback flow when generation fails."""
        return {
            "name": flow_name or "Basic Chat Flow",
            "description": f"Simple flow for: {user_request}",
            "data": self._create_basic_fallback_flow_data(user_request),
            "is_component": False,
        }
    
    def _create_basic_fallback_flow_data(self, user_request: str) -> Dict[str, Any]:
        """Create basic flow data for fallback scenarios."""
        # Prefer local catalog via RAG to get real templates
        try:
            if not self.rag.components_cache:
                self.rag._load_langflow_components()
            cache = self.rag.components_cache or {}
            chat_input = None
            openai_comp = None
            chat_output = None
            for category, comps in cache.items():
                if not isinstance(comps, dict):
                    continue
                chat_input = chat_input or comps.get("ChatInput")
                chat_output = chat_output or comps.get("ChatOutput")
                # Prefer OpenAIModel key name
                if "OpenAIModel" in comps:
                    openai_comp = comps.get("OpenAIModel")
                elif "OpenAI" in comps:
                    openai_comp = openai_comp or comps.get("OpenAI")
            if chat_input and openai_comp and chat_output:
                self.logger.info("📦 Using local catalog templates for fallback flow")
                return {
                    "nodes": [
                        {
                            "id": "ChatInput-1",
                            "type": "genericNode",
                            "position": {"x": 100, "y": 100},
                            "data": {
                                "type": "ChatInput",
                                "id": "ChatInput-1",
                                "node": chat_input
                            }
                        },
                        {
                            "id": "OpenAI-1",
                            "type": "genericNode",
                            "position": {"x": 400, "y": 100},
                            "data": {
                                "type": "OpenAIModel" if isinstance(openai_comp, dict) and openai_comp.get("display_name","OpenAI").startswith("OpenAI") else "OpenAI",
                                "id": "OpenAI-1",
                                "node": openai_comp
                            }
                        },
                        {
                            "id": "ChatOutput-1",
                            "type": "genericNode",
                            "position": {"x": 700, "y": 100},
                            "data": {
                                "type": "ChatOutput",
                                "id": "ChatOutput-1",
                                "node": chat_output
                            }
                        }
                    ],
                    "edges": [
                        {
                            "id": "reactflow__edge-ChatInput-1-OpenAI-1",
                            "source": "ChatInput-1",
                            "target": "OpenAI-1",
                            "sourceHandle": '{"id":"ChatInput-1","dataType":"Message","name":"message","output_types":["Message"]}',
                            "targetHandle": '{"type":"Message","fieldName":"input_value","id":"OpenAI-1","inputTypes":["Message"]}'
                        },
                        {
                            "id": "reactflow__edge-OpenAI-1-ChatOutput-1",
                            "source": "OpenAI-1",
                            "target": "ChatOutput-1",
                            "sourceHandle": '{"id":"OpenAI-1","dataType":"Message","name":"text_output","output_types":["Message"]}',
                            "targetHandle": '{"type":"Message","fieldName":"input_value","id":"ChatOutput-1","inputTypes":["Message"]}'
                        }
                    ],
                    "viewport": {"x": 0, "y": 0, "zoom": 1}
                }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not build fallback from local catalog: {e}")
        # Fallback to minimal structure with basic templates
        self.logger.info("📦 Using minimal fallback flow structure")
        return {
            "nodes": [
                {
                    "id": "ChatInput-1",
                    "type": "genericNode",
                    "position": {"x": 100, "y": 100},
                    "data": {
                        "type": "ChatInput",
                        "id": "ChatInput-1",
                        "node": {
                            "display_name": "Chat Input",
                            "description": "Get chat inputs from the user",
                            "template": {
                                "input_value": {
                                    "type": "str",
                                    "required": False,
                                    "placeholder": "",
                                    "show": True,
                                    "advanced": False,
                                    "name": "input_value",
                                    "display_name": "Input Value"
                                }
                            },
                            "base_classes": ["Message"],
                            "outputs": [{"name": "message", "types": ["Message"], "display_name": "Message"}]
                        }
                    }
                },
                {
                    "id": "OpenAI-1", 
                    "type": "genericNode",
                    "position": {"x": 400, "y": 100},
                    "data": {
                        "type": "OpenAI",
                        "id": "OpenAI-1",
                        "node": {
                            "display_name": "OpenAI",
                            "description": "OpenAI language model",
                            "template": {
                                "input_value": {
                                    "type": "Message",
                                    "required": False,
                                    "placeholder": "",
                                    "show": True,
                                    "advanced": False,
                                    "name": "input_value",
                                    "display_name": "Input",
                                    "input_types": ["Message"]
                                }
                            },
                            "base_classes": ["LanguageModel"],
                            "outputs": [{"name": "text_output", "types": ["Message"], "display_name": "Text"}]
                        }
                    }
                },
                {
                    "id": "ChatOutput-1",
                    "type": "genericNode", 
                    "position": {"x": 700, "y": 100},
                    "data": {
                        "type": "ChatOutput",
                        "id": "ChatOutput-1",
                        "node": {
                            "display_name": "Chat Output",
                            "description": "Display chat output",
                            "template": {
                                "input_value": {
                                    "type": "Message",
                                    "required": True,
                                    "placeholder": "",
                                    "show": True,
                                    "advanced": False,
                                    "name": "input_value",
                                    "display_name": "Input",
                                    "input_types": ["Message"]
                                }
                            },
                            "base_classes": ["Message"]
                        }
                    }
                }
            ],
            "edges": [
                {
                    "id": "reactflow__edge-ChatInput-1-OpenAI-1",
                    "source": "ChatInput-1",
                    "target": "OpenAI-1",
                    "sourceHandle": '{"id":"ChatInput-1","dataType":"Message","name":"message","output_types":["Message"]}',
                    "targetHandle": '{"type":"Message","fieldName":"input_value","id":"OpenAI-1","inputTypes":["Message"]}'
                },
                {
                    "id": "reactflow__edge-OpenAI-1-ChatOutput-1", 
                    "source": "OpenAI-1",
                    "target": "ChatOutput-1",
                    "sourceHandle": '{"id":"OpenAI-1","dataType":"Message","name":"text_output","output_types":["Message"]}',
                    "targetHandle": '{"type":"Message","fieldName":"input_value","id":"ChatOutput-1","inputTypes":["Message"]}'
                }
            ],
            "viewport": {"x": 0, "y": 0, "zoom": 1}
        }
    
    def validate_flow_json(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a flow data structure."""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Check required keys
            if "nodes" not in flow_data:
                errors.append("Flow data must contain 'nodes' array")
            elif not flow_data["nodes"]:
                errors.append("Flow must contain at least one node")
            
            if "edges" not in flow_data:
                errors.append("Flow data must contain 'edges' array")
            
            # Check node structure
            if "nodes" in flow_data:
                for i, node in enumerate(flow_data["nodes"]):
                    if "id" not in node:
                        errors.append(f"Node {i} missing required 'id' field")
                    if "data" not in node or "type" not in node.get("data", {}):
                        errors.append(f"Node {i} missing required data.type field")
            
            # Check edges
            if "nodes" in flow_data and "edges" in flow_data:
                node_ids = {node.get("id") for node in flow_data["nodes"]}
                for i, edge in enumerate(flow_data["edges"]):
                    if edge.get("source") not in node_ids:
                        errors.append(f"Edge {i} references non-existent source node")
                    if edge.get("target") not in node_ids:
                        errors.append(f"Edge {i} references non-existent target node")
            
            # Warnings
            if len(flow_data.get("nodes", [])) > 1 and not flow_data.get("edges"):
                warnings.append("Multi-node flow without edges may not function properly")
            
            # Check for common components
            node_types = [node.get("data", {}).get("type") for node in flow_data.get("nodes", [])]
            if "ChatInput" not in node_types:
                suggestions.append("Consider adding a ChatInput component for user interaction")
            if "ChatOutput" not in node_types:
                suggestions.append("Consider adding a ChatOutput component to display results")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "suggestions": []
            }


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize agent
    agent = SimpleFlowBuilderAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test with educational example
    user_request = "Create a chatbot that answers questions about course syllabus using uploaded files"
    
    try:
        flow = agent.build_flow(user_request)
        
        print("Generated Flow:")
        print(f"Name: {flow['name']}")
        print(f"Description: {flow['description']}")
        print(f"Nodes: {len(flow['data']['nodes'])}")
        print(f"Edges: {len(flow['data']['edges'])}")
        
        # Validate the flow
        validation = agent.validate_flow_json(flow['data'])
        print(f"\nValidation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
        
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']: 
            print(f"Warnings: {validation['warnings']}")
        if validation['suggestions']:
            print(f"Suggestions: {validation['suggestions']}")
        
        # Save flow JSON
        with open("simple_generated_flow.json", "w", encoding="utf-8") as f:
            json.dump(flow, f, indent=2, ensure_ascii=False)
        
        print("\nFlow saved to simple_generated_flow.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
