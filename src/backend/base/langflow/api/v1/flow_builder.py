"""
Flow Builder Agent integration for Langflow API v1
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import os

from .schemas import FlowBuildRequest, FlowBuildResponse, FlowValidationRequest, FlowValidationResponse, ComponentSearchRequest, ComponentSearchResponse
# NEW: DB/session + models utilities for saving flows
from langflow.api.utils import DbSession, CurrentActiveUser
from langflow.services.database.models.flow.model import FlowCreate as DBFlowCreate
from .flows import _new_flow, _save_flow_to_fs

# Import our Flow Builder Agent
try:
    from flow_builder_agent.simple_agent import SimpleFlowBuilderAgent
    FLOW_BUILDER_AVAILABLE = True
except ImportError:
    FLOW_BUILDER_AVAILABLE = False
    SimpleFlowBuilderAgent = None

# Import Langflow component registry using backend shim
try:
    from langflow.interface.components import get_type_dict  # local shim re-exports from lfx
    LANGFLOW_TYPES_AVAILABLE = True
except Exception:
    LANGFLOW_TYPES_AVAILABLE = False
    get_type_dict = None


# Initialize router
router = APIRouter(prefix="/flow_builder", tags=["Flow Builder Agent"])

# Global agent instance
_agent = None


def get_flow_builder_agent():
    """Dependency to get the Flow Builder Agent instance.
    Chooses provider based on available API keys. Falls back to OpenAI if
    Gemini initialization fails (e.g., missing google-generativeai).
    """
    global _agent
    if not FLOW_BUILDER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Flow Builder Agent not available. Please install the flow_builder_agent package.",
        )

    if _agent is not None:
        return _agent

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Prefer Gemini when key is present
    if gemini_key:
        logging.info("FlowBuilderAgent: Initializing Gemini provider...")
        try:
            _agent = SimpleFlowBuilderAgent(
                api_key=gemini_key, provider="gemini", model_name=os.getenv("FLOW_BUILDER_MODEL", "gemini-2.5-flash")
            )
            logging.info("FlowBuilderAgent: Gemini initialized successfully")
            return _agent
        except Exception as e:
            logging.error(f"FlowBuilderAgent: Gemini init failed: {e}")
            if openai_key:
                logging.info("FlowBuilderAgent: Falling back to OpenAI provider...")
                try:
                    _agent = SimpleFlowBuilderAgent(openai_api_key=openai_key)
                    logging.info("FlowBuilderAgent: OpenAI initialized successfully (fallback)")
                    return _agent
                except Exception as oe:
                    logging.error(f"FlowBuilderAgent: OpenAI fallback also failed: {oe}")
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Failed to initialize Gemini (" + str(e) + ") and OpenAI fallback (" + str(oe) + "). "
                            "Install google-generativeai or set OPENAI_API_KEY to use OpenAI."
                        ),
                    )
            # No OpenAI key to fall back to
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to initialize Gemini provider: "
                    + str(e)
                    + ". Install google-generativeai or unset GEMINI_API_KEY to avoid selecting Gemini."
                ),
            )

    # If no Gemini, try OpenAI
    if openai_key:
        logging.info("FlowBuilderAgent: Initializing OpenAI provider...")
        try:
            _agent = SimpleFlowBuilderAgent(openai_api_key=openai_key)
            logging.info("FlowBuilderAgent: OpenAI initialized successfully")
            return _agent
        except Exception as e:
            logging.error(f"FlowBuilderAgent: OpenAI init failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI provider: {e}")

    # Neither key provided
    raise HTTPException(
        status_code=500,
        detail=(
            "No AI provider API key configured. Set GEMINI_API_KEY (requires google-generativeai) "
            "or OPENAI_API_KEY in the backend environment."
        ),
    )


def enrich_nodes_with_templates(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich nodes with proper component templates from Langflow registry.
    This ensures nodes have the proper template structure that the frontend expects.
    """
    if not LANGFLOW_TYPES_AVAILABLE:
        logging.warning("⚠️ Langflow types not available, skipping node enrichment")
        return flow_data
    
    try:
        # Get all available component types
        all_types = get_type_dict()
        logging.info(f"📦 Fetched {len(all_types)} component categories from Langflow")
        
        # Enrich each node
        for node in flow_data.get("data", {}).get("nodes", []):
            node_type = node.get("data", {}).get("type")
            if not node_type:
                continue
            
            # Check if node.data.node already has template
            if "node" in node.get("data", {}) and "template" in node["data"]["node"]:
                logging.info(f"✅ Node {node['id']} already has template, skipping")
                continue
            
            # Search for component in all categories
            component_data = None
            for category, components in all_types.items():
                if node_type in components:
                    component_data = components[node_type]
                    logging.info(f"✅ Found template for {node_type} in category {category}")
                    break
            
            if component_data:
                # Ensure node.data.node exists and has proper structure
                if "node" not in node["data"]:
                    node["data"]["node"] = component_data
                else:
                    # Merge component data into existing node data
                    node["data"]["node"].update(component_data)
                
                logging.info(f"✅ Enriched node {node['id']} with template from {node_type}")
            else:
                logging.warning(f"⚠️ Could not find template for component type: {node_type}")
        
        logging.info("✅ Node enrichment complete")
        return flow_data
        
    except Exception as e:
        logging.error(f"❌ Error enriching nodes: {e}")
        return flow_data


@router.post("/build", response_model=FlowBuildResponse)
async def build_flow(
    *,
    session: DbSession,
    request: FlowBuildRequest,
    current_user: CurrentActiveUser,
    background_tasks: BackgroundTasks,
    agent = Depends(get_flow_builder_agent),
):
    """
    Build a Langflow flow from natural language description.
    If request.persist is True, the generated flow is saved so it appears in the Langflow UI.
    Otherwise we just return the flow JSON for the current canvas.
    """
    logging.info(f"🌐 API ENDPOINT: Received request - query: '{request.query}', flow_name: '{request.flow_name}', persist={getattr(request, 'persist', False)}")
    start_time = datetime.utcnow()
    try:
        logging.info(f"🤖 API ENDPOINT: Calling agent.build_flow_async...")
        # Generate the flow using the simplified agent
        flow_data = await agent.build_flow_async(
            user_request=request.query, 
            flow_name=request.flow_name
        )
        logging.info(f"✅ API ENDPOINT: Agent returned flow data")

        # Enrich nodes with proper templates from component registry
        logging.info(f"📦 API ENDPOINT: Enriching nodes with templates...")
        flow_data = enrich_nodes_with_templates(flow_data)
        logging.info(f"✅ API ENDPOINT: Node enrichment complete")

        flow_id: Optional[str] = None
        # Persist only if explicitly requested
        if getattr(request, "persist", False):
            try:
                db_flow_in = DBFlowCreate(
                    name=flow_data.get("name") or (request.flow_name or "Generated Flow"),
                    description=flow_data.get("description") or f"Generated from query: {request.query[:80]}",
                    data=flow_data.get("data") or {},
                    is_component=False,
                )
                db_flow = await _new_flow(session=session, flow=db_flow_in, user_id=current_user.id)
                await session.commit()
                await session.refresh(db_flow)
                # Save to filesystem if configured
                await _save_flow_to_fs(db_flow)
                flow_id = str(db_flow.id)
                logging.info(f"💾 API ENDPOINT: Flow persisted with id={flow_id}")
            except Exception as save_err:
                logging.error(f"⚠️ API ENDPOINT: Failed to persist generated flow: {save_err}")
                flow_id = None
        else:
            logging.info("📝 API ENDPOINT: Persistence disabled (persist=false), returning ephemeral flow JSON")

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Extract component types from the flow nodes
        components_used = []
        if flow_data.get("data", {}).get("nodes"):
            components_used = [
                node.get("data", {}).get("type", "Unknown") 
                for node in flow_data["data"]["nodes"]
            ]
        
        logging.info(f"🎉 API ENDPOINT: Success! {len(components_used)} components, {processing_time_ms}ms")
        
        return FlowBuildResponse(
            success=True,
            flow_id=flow_id,
            flow_json=flow_data,
            message="Flow generated successfully",
            timestamp=end_time,
            processing_time_ms=processing_time_ms,
            components_used=components_used,
            recommendations=[]
        )
            
    except Exception as e:
        logging.error(f"❌ API ENDPOINT ERROR: {str(e)}")
        logging.exception("Full traceback:")
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/validate", response_model=FlowValidationResponse)
async def validate_flow(
    request: FlowValidationRequest,
    agent = Depends(get_flow_builder_agent)
):
    """
    Validate a Langflow JSON structure.
    
    Checks if the provided JSON is a valid Langflow flow and provides suggestions.
    """
    try:
        validation_result = agent.validate_flow_json(request.flow_json)
        
        return FlowValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result.get("errors", []),
            warnings=validation_result.get("warnings", []),
            suggestions=validation_result.get("suggestions", [])
        )
        
    except Exception as e:
        logging.error(f"Error validating flow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )


@router.post("/components/search", response_model=ComponentSearchResponse)
async def search_components(
    request: ComponentSearchRequest,
    agent = Depends(get_flow_builder_agent)
):
    """
    Search for available Langflow components based on a query.
    
    Useful for discovering what components are available for specific tasks.
    """
    start_time = datetime.utcnow()
    
    try:
        # Use the RAG system to search for components
        results = agent.rag.search_components(
            query=request.query,
            top_k=request.limit
        )
        
        end_time = datetime.utcnow()
        query_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return ComponentSearchResponse(
            components=results,
            total_found=len(results),
            query_time_ms=query_time_ms
        )
        
    except Exception as e:
        logging.error(f"Error searching components: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


@router.get("/components/list")
async def list_all_components(agent = Depends(get_flow_builder_agent)):
    """
    List all available components in the knowledge base.
    
    Returns a comprehensive list of all components that can be used in flows.
    """
    try:
        components = agent.rag.get_all_components()
        
        return {
            "components": components,
            "total_count": len(components),
            "categories": list(set(comp.get("category", "unknown") for comp in components))
        }
        
    except Exception as e:
        logging.error(f"Error listing components: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving components: {str(e)}"
        )


@router.get("/examples")
async def get_example_flows():
    """
    Get example flows that demonstrate different patterns.
    
    Useful for understanding what the Flow Builder Agent can create.
    """
    try:
        # Return example flows from the knowledge base
        examples = [
            {
                "name": "Syllabus Q&A with Approval",
                "description": "Educational flow for answering questions about course syllabus with human approval",
                "complexity": "medium",
                "use_case": "education",
                "components": ["ChatInput", "OpenAI", "SyllabusLoader", "ChromaDB", "HumanApproval"]
            },
            {
                "name": "Simple Chat Bot", 
                "description": "Basic conversational AI agent",
                "complexity": "simple",
                "use_case": "general",
                "components": ["ChatInput", "OpenAI", "ChatOutput"]
            },
            {
                "name": "Document Q&A System",
                "description": "Upload documents and ask questions about them",
                "complexity": "medium", 
                "use_case": "document_analysis",
                "components": ["FileLoader", "TextSplitter", "VectorStore", "OpenAI", "ChatInput"]
            }
        ]
        
        return {
            "examples": examples,
            "total_count": len(examples)
        }
        
    except Exception as e:
        logging.error(f"Error getting examples: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving examples: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for the Flow Builder Agent."""
    try:
        if not FLOW_BUILDER_AVAILABLE:
            return {
                "status": "unavailable",
                "service": "Flow Builder Agent",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Flow Builder Agent package not installed"
            }
            
        # Test that we can initialize the agent
        agent = get_flow_builder_agent()
        
        return {
            "status": "healthy",
            "service": "Flow Builder Agent",
            "timestamp": datetime.utcnow().isoformat(),
            "model": agent.model_name,
            "rag_status": "active"
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )
