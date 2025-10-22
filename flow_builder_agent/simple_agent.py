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

from openai import AsyncOpenAI, OpenAI
import google.generativeai as genai

from .rag.component_rag import ComponentRAG


class SimpleFlowBuilderAgent:
    """
    Simplified Flow Builder Agent that generates basic Langflow flow structures.
    
    This agent creates flow data dictionaries that can be used with existing
    Langflow FlowCreate models for database operations.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None, 
                 provider: str = "gemini", openai_api_key: str = None):
        """
        Initialize the Simple Flow Builder Agent.
        
        Args:
            api_key: API key for the chosen provider (Gemini or OpenAI)
            model_name: Model to use (default: "gemini-2.5-flash" for Gemini, "gpt-4o" for OpenAI)
            provider: Either "gemini" or "openai" (default: "gemini")            openai_api_key: Backwards compatibility - if provided, uses OpenAI
        """
        self.rag = ComponentRAG()
        self.logger = logging.getLogger(__name__)
        
        # Handle backwards compatibility
        if openai_api_key:
            provider = "openai"
            api_key = openai_api_key
            model_name = model_name or "gpt-4o"
        
        self.provider = provider
        
        if provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.async_client = None
        elif provider == "openai":
            self.model_name = model_name or "gpt-4o"
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'")
    
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
                # Use Gemini API
                self.logger.info("🤖 CHECKPOINT 4b: Calling Gemini API...")
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
    "type": "EXACT_COMPONENT_NAME",  // USE THE EXACT COMPONENT NAME FROM THE LIST BELOW
    "id": "same_as_node_id",
    "node": {{}}  // Will be filled by backend
  }}

CRITICAL: The "type" field in node.data MUST be the EXACT component name from the available components list below.

Create a Langflow flow for: {user_request}

{component_info}

Space nodes horizontally: Input components at x=100-200, Processing at x=400-600, Output at x=800-900
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
            return self._validate_flow_data(flow_data)
            
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
        # Import requests to fetch component templates
        try:
            import requests
            # Try to fetch real component templates from Langflow backend
            response = requests.get("http://127.0.0.1:7860/api/v1/all", timeout=5)
            if response.status_code == 200:
                components_data = response.json()
                self.logger.info("✅ Successfully fetched component templates from Langflow")
                
                # Extract templates for our needed components
                chat_input_template = None
                openai_template = None
                chat_output_template = None
                
                # Search for components in the response
                for category, components in components_data.items():
                    if isinstance(components, dict):
                        for comp_name, comp_data in components.items():
                            if comp_name == "ChatInput":
                                chat_input_template = comp_data
                            elif comp_name in ["OpenAIModel", "OpenAI"]:
                                openai_template = comp_data
                            elif comp_name == "ChatOutput":
                                chat_output_template = comp_data
                
                # If we have the templates, use them
                if chat_input_template and openai_template and chat_output_template:
                    self.logger.info("📦 Using real component templates for fallback flow")
                    return {
                        "nodes": [
                            {
                                "id": "ChatInput-1",
                                "type": "genericNode",
                                "position": {"x": 100, "y": 100},
                                "data": {
                                    "type": "ChatInput",
                                    "id": "ChatInput-1",
                                    "node": chat_input_template
                                }
                            },
                            {
                                "id": "OpenAI-1", 
                                "type": "genericNode",
                                "position": {"x": 400, "y": 100},
                                "data": {
                                    "type": "OpenAIModel" if "OpenAIModel" in str(openai_template.get("display_name", "")) else "OpenAI",
                                    "id": "OpenAI-1",
                                    "node": openai_template
                                }
                            },
                            {
                                "id": "ChatOutput-1",
                                "type": "genericNode", 
                                "position": {"x": 700, "y": 100},
                                "data": {
                                    "type": "ChatOutput",
                                    "id": "ChatOutput-1",
                                    "node": chat_output_template
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
            self.logger.warning(f"⚠️ Could not fetch component templates: {e}")
        
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
        with open("simple_generated_flow.json", "w") as f:
            json.dump(flow, f, indent=2)
        
        print("\nFlow saved to simple_generated_flow.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
