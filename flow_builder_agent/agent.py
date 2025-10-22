"""
Flow Builder Agent - The main component that converts natural language requests 
into deployable Langflow JSON flows using RAG and structured LLM output.
"""

import json
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError
import google.generativeai as genai

# Use existing Langflow models instead of custom schemas
try:
    from langflow.services.database.models.flow.model import FlowCreate
    from langflow.api.v1.schemas import FlowBuildRequest, FlowBuildResponse
    LANGFLOW_MODELS_AVAILABLE = True
except ImportError:
    LANGFLOW_MODELS_AVAILABLE = False
    # Fallback for development/testing
    from typing import Dict, Any
    class FlowCreate:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from .rag.component_rag import ComponentRAG


class FlowBuilderAgent:
    """
    Meta-AI agent that converts natural language requests into deployable Langflow flows.
    
    This agent combines RAG (Retrieval-Augmented Generation) with structured output
    to ensure 100% reliable JSON generation for downstream machine consumption.
    
    Supports both OpenAI and Google Gemini models.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash", 
                provider: str = "gemini", openai_api_key: str = None):
        """
        Initialize the Flow Builder Agent.
        
        Args:
            api_key: API key for the chosen provider (Gemini or OpenAI)
            model_name: Model to use (default: "gemini-2.5-flash" for Gemini, "gpt-4o" for OpenAI)
            provider: Either "gemini" or "openai" (default: "gemini")
            openai_api_key: Backwards compatibility - if provided, uses OpenAI
        """
        self.provider = provider
        self.model_name = model_name
        self.rag = ComponentRAG()
        self.logger = logging.getLogger(__name__)
        
        # Handle backwards compatibility
        if openai_api_key:
            self.provider = "openai"
            api_key = openai_api_key
            if model_name == "gemini-2.5-flash":  # If using default Gemini model
                self.model_name = "gpt-4o"
        
        # Initialize the appropriate client
        if self.provider == "gemini":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.async_client = None  # Gemini async client if needed
        elif self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'")
        
        # Load system prompt templates
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load the system prompt for the LLM."""
        return """You are a Flow Builder Agent, an expert AI system that converts natural language requests into valid Langflow JSON flows.

Your mission is to:
1. Understand user requirements for AI workflows
2. Select appropriate Langflow components from the knowledge base
3. Generate a complete, valid Langflow flow JSON that can be directly deployed

CRITICAL REQUIREMENTS:
- Output must be 100% valid JSON that matches the Langflow schema exactly
- All component names, types, and connections must be accurate
- Every node must have a unique ID and proper positioning
- All edges must connect valid input/output handles
- Use ONLY components provided in the knowledge base

You will receive:
1. User's natural language request
2. Relevant components from RAG search
3. Suggested flow patterns
4. Component compatibility information

Generate a complete LangflowFlow object that implements the user's requirements."""

    def build_flow(self, user_request: str, **kwargs) -> FlowGenerationResponse:
        """
        Main method to build a complete Langflow flow from user request.
        
        Args:
            user_request: Natural language description of desired workflow
            **kwargs: Additional parameters for customization
            
        Returns:
            Complete flow generation response with JSON and explanation
        """
        try:
            # Parse user request
            request = UserRequest(request=user_request)
            
            # Analyze request using RAG
            analysis = self.rag.analyze_user_request(request)
            
            # Get relevant components and patterns
            relevant_components = self.rag.search_components(user_request, top_k=10)
            relevant_patterns = self.rag.search_patterns(user_request, top_k=3)
            
            # Generate flow using LLM with structured output
            flow = self._generate_flow_with_llm(
                request, analysis, relevant_components, relevant_patterns
            )
            
            # Validate and enhance the generated flow
            validated_flow = self._validate_and_enhance_flow(flow)
            
            # Generate explanation
            explanation = self._generate_explanation(validated_flow, analysis)
            
            # Create response
            return FlowGenerationResponse(
                flow=validated_flow,
                explanation=explanation,
                components_used=[node.data.type for node in validated_flow.data.nodes],
                reasoning=self._generate_reasoning(analysis, relevant_components),
                warnings=self._check_for_warnings(validated_flow)
            )
            
        except Exception as e:
            self.logger.error(f"Error building flow: {str(e)}")
            raise
    
    async def build_flow_async(self, user_request: str, **kwargs) -> FlowGenerationResponse:
        """
        Async version of the main method to build a complete Langflow flow from user request.
        
        Args:
            user_request: Natural language description of desired workflow
            **kwargs: Additional parameters for customization
            
        Returns:
            Complete flow generation response with JSON and explanation
        """
        try:
            # Parse user request
            request = UserRequest(request=user_request)
            
            # Analyze request using RAG
            analysis = self.rag.analyze_user_request(request)
            
            # Get relevant components and patterns
            relevant_components = self.rag.search_components(user_request, top_k=10)
            relevant_patterns = self.rag.search_patterns(user_request, top_k=3)
            
            # Generate flow using LLM with structured output
            flow = await self._generate_flow_with_llm_async(
                request, analysis, relevant_components, relevant_patterns
            )
            
            # Validate and enhance the generated flow
            validated_flow = self._validate_and_enhance_flow(flow)
            
            # Generate explanation
            explanation = self._generate_explanation(validated_flow, analysis)
            
            # Create response
            return FlowGenerationResponse(
                flow=validated_flow,
                explanation=explanation,
                components_used=[node.data.type for node in validated_flow.data.nodes],
                reasoning=self._generate_reasoning(analysis, relevant_components),
                warnings=self._check_for_warnings(validated_flow)
            )
            
        except Exception as e:
            self.logger.error(f"Error building flow: {str(e)}")
            raise

    async def generate_flow_async(self, request: FlowGenerationRequest) -> FlowGenerationResponse:
        """Async version of build_flow for API usage."""
        try:
            # Convert to the expected format and call build_flow in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.build_flow,
                request.user_request.query,
                {"flow_name": request.flow_name, "complexity": request.complexity}
            )
            
            return FlowGenerationResponse(
                success=True,
                flow_id=str(uuid.uuid4()),
                flow_json=result.flow.dict() if hasattr(result, 'flow') else None,
                components_used=result.components_used if hasattr(result, 'components_used') else [],
                recommendations=result.warnings if hasattr(result, 'warnings') else [],
                error_message=None
            )
            
        except Exception as e:
            return FlowGenerationResponse(
                success=False,
                flow_id=None,
                flow_json=None,
                components_used=[],
                recommendations=[],
                error_message=str(e)
            )
    
    def validate_flow_json(self, flow_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a Langflow JSON structure."""
        try:
            # Try to parse with Pydantic schema
            flow = LangflowFlow.parse_obj(flow_json)
            
            errors = []
            warnings = []
            suggestions = []
            
            # Basic validation checks
            if not flow.data.nodes:
                errors.append("Flow must contain at least one node")
                
            if len(flow.data.nodes) > 1 and not flow.data.edges:
                warnings.append("Multi-node flow without edges may not function properly")
            
            # Check for common patterns
            input_nodes = [n for n in flow.data.nodes if "input" in n.data.type.lower()]
            output_nodes = [n for n in flow.data.nodes if "output" in n.data.type.lower()]
            
            if not input_nodes:
                warnings.append("No input nodes found - consider adding ChatInput or FileInput")
                
            if not output_nodes:
                warnings.append("No output nodes found - consider adding ChatOutput or TextOutput")
            
            # Check node positions
            for node in flow.data.nodes:
                if node.position.x < 0 or node.position.y < 0:
                    suggestions.append(f"Node {node.id} has negative position - consider repositioning")
            return {
                "valid": True,
                "errors": errors,
                "warnings": warnings, 
                "suggestions": suggestions
            }
            
        except ValidationError as e:
            return {
                "valid": False,
                "errors": [f"Schema validation error: {str(e)}"],
                "warnings": [],
                "suggestions": ["Check the Langflow JSON structure against the schema"]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Unexpected validation error: {str(e)}"],
                "warnings": [],
                "suggestions": []
            }
    
    def _generate_flow_with_llm(
        self, 
        request: UserRequest,
        analysis: Dict[str, Any],
        relevant_components: List[Tuple[str, Any, float]], 
        relevant_patterns: List[Tuple[str, Any, float]]
    ) -> LangflowFlow:
        """Generate the flow using LLM with structured output."""
        
        # Prepare context for LLM
        context = self._prepare_llm_context(analysis, relevant_components, relevant_patterns)
        
        # Create the prompt
        prompt = self._create_generation_prompt(request.request, context)
        
        if self.provider == "gemini":
            return self._generate_with_gemini(prompt)
        else:
            return self._generate_with_openai(prompt)
    
    def _generate_with_gemini(self, prompt: str) -> LangflowFlow:
        """Generate flow using Google Gemini."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}\n\nGenerate a complete Langflow flow JSON."
        
        # Use Gemini's generate_content with JSON mode
        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse the JSON response
        flow_data = json.loads(response.text)
        
        # Convert to Pydantic model
        return self._convert_llm_output_to_flow(flow_data)
    
    def _generate_with_openai(self, prompt: str) -> LangflowFlow:
        """Generate flow using OpenAI."""
        # Call LLM with function calling for structured output
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "generate_langflow_flow",
                "description": "Generate a complete Langflow flow JSON",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the flow"},
                        "description": {"type": "string", "description": "Description of what the flow does"},
                        "nodes": {
                            "type": "array",
                            "description": "Array of nodes in the flow",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "position": {
                                        "type": "object", 
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"}
                                        }
                                    },
                                    "data": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "id": {"type": "string"},
                                            "node": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        },
                        "edges": {
                            "type": "array",
                            "description": "Array of edges connecting nodes",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "source": {"type": "string"}, 
                                    "target": {"type": "string"},
                                    "sourceHandle": {"type": "string"},
                                    "targetHandle": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["name", "description", "nodes", "edges"]
                }
            }],
            function_call={"name": "generate_langflow_flow"},
            temperature=0.1
        )
        
        # Parse the function call response
        function_call = response.choices[0].message.function_call
        flow_data = json.loads(function_call.arguments)
        
        # Convert to Pydantic model
        return self._convert_llm_output_to_flow(flow_data)
    
    async def _generate_flow_with_llm_async(
        self, 
        request: UserRequest,
        analysis: Dict[str, Any],
        relevant_components: List[Tuple[str, Any, float]], 
        relevant_patterns: List[Tuple[str, Any, float]]
    ) -> LangflowFlow:
        """Generate the flow using LLM with structured output asynchronously."""
        
        # Prepare context for LLM
        context = self._prepare_llm_context(analysis, relevant_components, relevant_patterns)
        
        # Create the prompt
        prompt = self._create_generation_prompt(request.request, context)
        
        # Call LLM with function calling for structured output
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "generate_langflow_flow",
                "description": "Generate a complete Langflow flow JSON",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the flow"},
                        "description": {"type": "string", "description": "Description of what the flow does"},
                        "nodes": {
                            "type": "array",
                            "description": "Array of nodes in the flow",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "position": {
                                        "type": "object", 
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"}
                                        }
                                    },
                                    "data": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "id": {"type": "string"},
                                            "node": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        },
                        "edges": {
                            "type": "array",
                            "description": "Array of edges connecting nodes",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "source": {"type": "string"}, 
                                    "target": {"type": "string"},
                                    "sourceHandle": {"type": "string"},
                                    "targetHandle": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["name", "description", "nodes", "edges"]
                }
            }],
            function_call={"name": "generate_langflow_flow"},
            temperature=0.1
        )
        
        # Parse the function call response
        function_call = response.choices[0].message.function_call
        flow_data = json.loads(function_call.arguments)
        
        # Convert to Pydantic model
        return self._convert_llm_output_to_flow(flow_data)
    
    def _prepare_llm_context(
        self,
        analysis: Dict[str, Any],
        relevant_components: List[Tuple[str, Any, float]],
        relevant_patterns: List[Tuple[str, Any, float]]
    ) -> Dict[str, Any]:
        """Prepare structured context for the LLM."""
        
        context = {
            "requirements": analysis["detected_requirements"],
            "complexity": analysis["estimated_complexity"],
            "available_components": {},
            "suggested_patterns": [],
            "component_compatibility": {}
        }
        
        # Add component details
        for comp_name, component, score in relevant_components:
            context["available_components"][comp_name] = {
                "display_name": component.display_name,
                "description": component.description,
                "category": component.category,
                "inputs": {
                    name: {
                        "type": field.type,
                        "required": field.required,
                        "description": field.info or f"Input for {name}"
                    } for name, field in component.inputs.items()
                },
                "outputs": [
                    {
                        "name": output.name,
                        "types": output.types,
                        "description": output.display_name
                    } for output in (component.outputs or [])
                ],
                "relevance_score": score
            }
            
            # Add compatibility information
            compatible = self.rag.get_compatible_components(comp_name)
            context["component_compatibility"][comp_name] = [
                {"name": name, "relationship": relationship} 
                for name, _, relationship in compatible[:5]  # Limit to top 5
            ]
        
        # Add pattern information
        for pattern_name, pattern, score in relevant_patterns:
            context["suggested_patterns"].append({
                "name": pattern.name,
                "description": pattern.description,
                "components": pattern.components,
                "connections": pattern.connections,
                "relevance_score": score
            })
        
        return context
    
    def _create_generation_prompt(self, user_request: str, context: Dict[str, Any]) -> str:
        """Create the detailed prompt for flow generation."""
        
        prompt = f"""
USER REQUEST: {user_request}

ANALYSIS:
- Requirements: {', '.join(context['requirements'])}
- Complexity: {context['complexity']}

AVAILABLE COMPONENTS:
"""
        
        for comp_name, comp_info in context["available_components"].items():
            prompt += f"""
{comp_name} ({comp_info['category']}):
- Description: {comp_info['description']}
- Inputs: {list(comp_info['inputs'].keys())}
- Outputs: {[out['name'] for out in comp_info['outputs']]}
"""
        
        if context["suggested_patterns"]:
            prompt += "\nSUGGESTED PATTERNS:\n"
            for pattern in context["suggested_patterns"]:
                prompt += f"- {pattern['name']}: {pattern['description']}\n"
                prompt += f"  Components: {', '.join(pattern['components'])}\n"
        
        prompt += """
GENERATION REQUIREMENTS:
1. Create a complete, valid Langflow flow that fulfills the user request
2. Use appropriate components from the available list
3. Ensure all nodes have unique IDs (use format: ComponentType-{random_5_chars})
4. Position nodes logically (inputs on left, outputs on right, processing in middle)
5. Create proper edges between compatible components
6. Set reasonable default values for component parameters
7. Include proper source and target handles for edges

Generate the flow now using the function call.
"""
        return prompt
    
    def _convert_llm_output_to_flow(self, flow_data: Dict[str, Any]) -> LangflowFlow:
        """Convert LLM output to proper Pydantic flow model."""
        
        # Create nodes
        nodes = []
        for node_data in flow_data.get("nodes", []):
            node_id = node_data["id"]
            node_type = node_data["data"]["type"]
            
            # Get component definition
            component = ALL_COMPONENTS.get(node_type)
            if not component:
                raise ValueError(f"Unknown component type: {node_type}")
            
            # Create node data
            api_class = APIClassType(
                description=component.description,
                display_name=component.display_name,
                template={name: field for name, field in component.inputs.items()},
                documentation=component.documentation,
                base_classes=component.base_classes,
                outputs=component.outputs or []
            )
            
            node_data_obj = NodeDataType(
                type=node_type,
                node=api_class,
                id=node_id
            )
            
            node = LangflowNode(
                id=node_id,
                type="genericNode",
                position=NodePosition(**node_data["position"]),
                data=node_data_obj
            )
            nodes.append(node)
        
        # Create edges
        edges = []
        for edge_data in flow_data.get("edges", []):
            edge = LangflowEdge(
                id=edge_data.get("id", f"edge-{uuid.uuid4().hex[:8]}"),
                source=edge_data["source"],
                target=edge_data["target"],
                sourceHandle=edge_data.get("sourceHandle"),
                targetHandle=edge_data.get("targetHandle")
            )
            edges.append(edge)
        
        # Create flow
        flow = LangflowFlow(
            name=flow_data["name"],
            description=flow_data["description"],
            data=FlowData(
                nodes=nodes,
                edges=edges,
                viewport=Viewport()
            )
        )
        
        return flow
    
    def _validate_and_enhance_flow(self, flow: LangflowFlow) -> LangflowFlow:
        """Validate the generated flow and enhance it with proper handles and IDs."""
        
        # Ensure all nodes have unique IDs
        used_ids = set()
        for node in flow.data.nodes:
            if node.id in used_ids:
                node.id = f"{node.data.type}-{uuid.uuid4().hex[:5]}"
            used_ids.add(node.id)
        
        # Generate proper edge IDs and handles if missing
        for i, edge in enumerate(flow.data.edges):
            if not edge.id:
                edge.id = f"reactflow__edge-{edge.source}-{edge.target}-{i}"
            
            # Add basic handles if missing (simplified version)
            if not edge.sourceHandle:
                edge.sourceHandle = f'{{\"id\":\"{edge.source}\",\"dataType\":\"str\",\"name\":\"output\",\"output_types\":[\"str\"]}}'
            
            if not edge.targetHandle:
                edge.targetHandle = f'{{\"id\":\"{edge.target}\",\"type\":\"str\",\"fieldName\":\"input\",\"name\":\"input\"}}'
        
        # Adjust node positions for better layout
        self._optimize_node_positions(flow.data.nodes)
        
        return flow
    
    def _optimize_node_positions(self, nodes: List[LangflowNode]):
        """Optimize node positions for a clean layout."""
        
        # Categorize nodes
        input_nodes = []
        processing_nodes = []
        output_nodes = []
        
        for node in nodes:
            if "Input" in node.data.type:
                input_nodes.append(node)
            elif "Output" in node.data.type:
                output_nodes.append(node)
            else:
                processing_nodes.append(node)
        
        # Position nodes in columns
        y_spacing = 200
        x_spacing = 300
        
        # Input column (x=100)
        for i, node in enumerate(input_nodes):
            node.position.x = 100
            node.position.y = 100 + i * y_spacing
        
        # Processing columns (x=400, 700, etc.)
        processing_columns = max(1, len(processing_nodes) // 3 + 1)
        for i, node in enumerate(processing_nodes):
            col = i % processing_columns
            row = i // processing_columns
            node.position.x = 400 + col * x_spacing
            node.position.y = 100 + row * y_spacing
        
        # Output column (rightmost)
        output_x = 400 + processing_columns * x_spacing
        for i, node in enumerate(output_nodes):
            node.position.x = output_x
            node.position.y = 100 + i * y_spacing
    
    def _generate_explanation(self, flow: LangflowFlow, analysis: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the generated flow."""
        
        explanation = f"""Flow: {flow.name}

Description: {flow.description}

This flow addresses the following requirements:
"""
        
        for req in analysis["detected_requirements"]:
            explanation += f"- {req.replace('_', ' ').title()}\n"
        
        explanation += f"\nComponents used ({len(flow.data.nodes)} total):\n"
        
        for node in flow.data.nodes:
            component = ALL_COMPONENTS.get(node.data.type)
            if component:
                explanation += f"- {component.display_name}: {component.description}\n"
        
        explanation += f"\nConnections ({len(flow.data.edges)} total):\n"
        for edge in flow.data.edges:
            source_node = next((n for n in flow.data.nodes if n.id == edge.source), None)
            target_node = next((n for n in flow.data.nodes if n.id == edge.target), None)
            
            if source_node and target_node:
                source_comp = ALL_COMPONENTS.get(source_node.data.type)
                target_comp = ALL_COMPONENTS.get(target_node.data.type)
                
                if source_comp and target_comp:
                    explanation += f"- {source_comp.display_name} → {target_comp.display_name}\n"
        
        explanation += "\nThis flow can be directly imported into Langflow and deployed."
        
        return explanation
    
    def _generate_reasoning(self, analysis: Dict[str, Any], relevant_components: List[Tuple]) -> str:
        """Generate reasoning for component selection."""
        
        reasoning = "Component Selection Reasoning:\n\n"
        
        reasoning += f"Based on the analysis, detected requirements: {', '.join(analysis['detected_requirements'])}\n\n"
        
        reasoning += "Selected components based on:\n"
        for comp_name, component, score in relevant_components[:5]:
            reasoning += f"- {component.display_name}: Relevance score {score:.3f}, "
            reasoning += f"category: {component.category}\n"
        
        return reasoning
    
    def _check_for_warnings(self, flow: LangflowFlow) -> Optional[List[str]]:
        """Check for potential issues and return warnings."""
        
        warnings = []
        
        # Check for unconnected nodes
        connected_nodes = set()
        for edge in flow.data.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        for node in flow.data.nodes:
            if node.id not in connected_nodes and len(flow.data.nodes) > 1:
                warnings.append(f"Node {node.data.type} appears to be unconnected")
        
        # Check for missing API keys in OpenAI components
        for node in flow.data.nodes:
            if "OpenAI" in node.data.type:
                warnings.append("Remember to configure OpenAI API key for OpenAI components")
        
        # Check for file components
        for node in flow.data.nodes:
            if "File" in node.data.type:
                warnings.append("Remember to upload files for file input components")
        
        return warnings if warnings else None
    
    def deploy_flow(self, flow: LangflowFlow, langflow_api_url: str, auth_token: str) -> Dict[str, Any]:
        """
        Deploy the generated flow to a Langflow instance.
        
        Args:
            flow: The generated flow to deploy
            langflow_api_url: URL of the Langflow API
            auth_token: Authentication token for the API
            
        Returns:
            Deployment response with flow ID and status
        """
        import requests
        
        try:
            # Convert flow to JSON
            flow_json = flow.dict()
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json"
            }
            
            # POST to flows endpoint
            response = requests.post(
                f"{langflow_api_url}/api/v1/flows/",
                json=flow_json,
                headers=headers
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            self.logger.info(f"Successfully deployed flow: {result.get('id', 'unknown')}")
            
            return {
                "success": True,
                "flow_id": result.get("id"),
                "message": "Flow deployed successfully",
                "api_response": result
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to deploy flow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to deploy flow"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during deployment: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Unexpected deployment error"            }


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize agent
    agent = FlowBuilderAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test with the syllabus example
    user_request = """I want a bot to answer student questions about their due dates using only my syllabus file, and I need the draft response sent to my inbox for final approval before it is sent to the student."""
    
    try:
        response = agent.build_flow(user_request)
        
        print("Generated Flow:")
        print(f"Name: {response.flow.name}")
        print(f"Description: {response.flow.description}")
        print(f"Components: {', '.join(response.components_used)}")
        print(f"\nExplanation:\n{response.explanation}")
        
        # Save flow JSON
        flow_json = response.flow.dict()
        with open("generated_flow.json", "w") as f:
            json.dump(flow_json, f, indent=2)
        
        print("\nFlow saved to generated_flow.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
