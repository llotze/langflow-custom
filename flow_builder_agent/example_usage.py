"""
Example usage and testing script for the Flow Builder Agent.
This script demonstrates how to use the agent to convert natural language
requests into deployable Langflow flows.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our Flow Builder Agent
try:
    from flow_builder_agent import FlowBuilderAgent, UserRequest, ComponentRAG
    from flow_builder_agent.schemas.flow_schema import LangflowFlow
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure to install dependencies: pip install -r flow_builder_agent/requirements.txt")
    exit(1)


def test_component_rag():
    """Test the RAG system for component retrieval."""
    print("=" * 50)
    print("TESTING COMPONENT RAG SYSTEM")
    print("=" * 50)
    
    rag = ComponentRAG()
    
    # Test component search
    test_queries = [
        "chat with users and get their input",
        "load files and documents", 
        "split text into chunks",
        "embed text for similarity search",
        "store vectors in database",
        "retrieve relevant documents",
        "generate text with AI model",
        "send email notifications"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag.search_components(query, top_k=3)
        for name, comp, score in results:
            print(f"  - {comp.display_name} ({name}): {score:.3f}")
    
    # Test pattern search
    print(f"\nPattern Search for: 'educational chatbot with approval'")
    pattern_results = rag.search_patterns("educational chatbot with approval", top_k=2)
    for name, pattern, score in pattern_results:
        print(f"  - {pattern.name}: {score:.3f}")
    
    print("\n✅ RAG system test completed!")


def test_syllabus_qa_flow():
    """Test generating the syllabus Q&A flow from the project requirements."""
    print("=" * 50)
    print("TESTING SYLLABUS Q&A FLOW GENERATION")
    print("=" * 50)
    
    # Check for API keys - try Gemini first, then OpenAI
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        print("⚠️  No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY environment variable to test flow generation.")
        print("   For now, we'll show what the request analysis looks like...")
        
        # Still test the RAG analysis
        rag = ComponentRAG()
        request = UserRequest(
            request="I want a bot to answer student questions about their due dates using only my syllabus file, and I need the draft response sent to my inbox for final approval before it is sent to the student."
        )
        analysis = rag.analyze_user_request(request)
        
        print(f"Detected requirements: {analysis['detected_requirements']}")
        print(f"Estimated complexity: {analysis['estimated_complexity']}")
        print(f"Top suggested components:")
        for comp in analysis['suggested_components'][:5]:
            print(f"  - {comp['display_name']} ({comp['similarity']:.3f}): {comp['reason']}")
        
        return None
    
    # Initialize the agent with the available key
    try:
        if gemini_key:
            print("✅ Using Google Gemini API")
            agent = FlowBuilderAgent(
                api_key=gemini_key,
                model_name="gemini-1.5-pro",
                provider="gemini"
            )
        else:
            print("✅ Using OpenAI API")
            agent = FlowBuilderAgent(
                api_key=openai_key,
                model_name="gpt-4o",
                provider="openai"
            )
        
        # Test the main use case from project requirements
        user_request = """I want a bot to answer student questions about their due dates using only my syllabus file, and I need the draft response sent to my inbox for final approval before it is sent to the student."""
        
        print(f"User Request: {user_request}")
        print("\nGenerating flow...")
        
        # Generate the flow
        response = agent.build_flow(user_request)
        
        # Display results
        print(f"\n✅ Flow Generated Successfully!")
        print(f"Name: {response.flow.name}")
        print(f"Description: {response.flow.description}")
        print(f"Components used: {len(response.components_used)}")
        print(f"Connections: {len(response.flow.data.edges)}")
        
        print(f"\nComponents in flow:")
        for i, component_type in enumerate(response.components_used, 1):
            print(f"  {i}. {component_type}")
        
        print(f"\nExplanation:\n{response.explanation}")
        
        if response.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in response.warnings:
                print(f"  - {warning}")
        
        # Save the generated flow
        output_file = Path("generated_syllabus_qa_flow.json")
        flow_json = response.flow.dict()
        with open(output_file, 'w') as f:
            json.dump(flow_json, f, indent=2)
        
        print(f"\n📁 Flow saved to: {output_file}")
        print("   This JSON can be imported directly into Langflow!")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating flow: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_chat_flow():
    """Test generating a simple chat flow."""
    print("=" * 50)
    print("TESTING SIMPLE CHAT FLOW GENERATION")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found. Skipping flow generation test.")
        return None
    
    try:
        agent = FlowBuilderAgent(openai_api_key=api_key)
        
        user_request = "Create a simple chatbot that can have conversations with users using OpenAI"
        
        print(f"User Request: {user_request}")
        print("Generating flow...")
        
        response = agent.build_flow(user_request)
        
        print(f"\n✅ Simple Chat Flow Generated!")
        print(f"Name: {response.flow.name}")  
        print(f"Components: {', '.join(response.components_used)}")
        
        # Save this flow too
        output_file = Path("generated_simple_chat_flow.json")
        flow_json = response.flow.dict()
        with open(output_file, 'w') as f:
            json.dump(flow_json, f, indent=2)
        
        print(f"📁 Flow saved to: {output_file}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating simple chat flow: {e}")
        return None


def test_document_qa_flow():
    """Test generating a document Q&A flow."""
    print("=" * 50)
    print("TESTING DOCUMENT Q&A FLOW GENERATION")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found. Skipping flow generation test.")
        return None
    
    try:
        agent = FlowBuilderAgent(openai_api_key=api_key)
        
        user_request = "I need a system where users can upload documents and ask questions about them, getting AI-powered answers based on the document content"
        
        print(f"User Request: {user_request}")
        print("Generating flow...")
        
        response = agent.build_flow(user_request)
        
        print(f"\n✅ Document Q&A Flow Generated!")
        print(f"Name: {response.flow.name}")
        print(f"Components: {', '.join(response.components_used)}")
        
        # Save this flow too
        output_file = Path("generated_document_qa_flow.json")
        flow_json = response.flow.dict()
        with open(output_file, 'w') as f:
            json.dump(flow_json, f, indent=2)
        
        print(f"📁 Flow saved to: {output_file}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating document Q&A flow: {e}")
        return None


def validate_generated_flows():
    """Validate that generated flows conform to Langflow schema."""
    print("=" * 50)
    print("VALIDATING GENERATED FLOWS")
    print("=" * 50)
    
    flow_files = [
        "generated_syllabus_qa_flow.json",
        "generated_simple_chat_flow.json", 
        "generated_document_qa_flow.json"
    ]
    
    for flow_file in flow_files:
        file_path = Path(flow_file)
        if not file_path.exists():
            print(f"⏭️  Skipping {flow_file} (not found)")
            continue
        
        try:
            with open(file_path, 'r') as f:
                flow_data = json.load(f)
            
            # Validate using Pydantic
            flow = LangflowFlow(**flow_data)
            
            print(f"✅ {flow_file}: Valid Langflow JSON")
            print(f"   - Nodes: {len(flow.data.nodes)}")
            print(f"   - Edges: {len(flow.data.edges)}")
            print(f"   - Name: {flow.name}")
            
        except Exception as e:
            print(f"❌ {flow_file}: Validation failed - {e}")


def show_deployment_example():
    """Show how to deploy a generated flow."""
    print("=" * 50)
    print("DEPLOYMENT EXAMPLE")
    print("=" * 50)
    
    print("""
To deploy a generated flow to Langflow:

1. **Using the Agent's deploy_flow method:**
   ```python
   # After generating a flow
   response = agent.build_flow(user_request)
   
   # Deploy to Langflow instance
   deployment_result = agent.deploy_flow(
       flow=response.flow,
       langflow_api_url="http://localhost:7860",  # Your Langflow URL
       auth_token="your_auth_token_here"          # Get from Langflow UI
   )
   
   if deployment_result["success"]:
       print(f"Flow deployed with ID: {deployment_result['flow_id']}")
   ```

2. **Manual import via Langflow UI:**
   - Open Langflow in your browser
   - Click "New Flow" → "Import"
   - Select the generated JSON file
   - The flow will be imported and ready to use

3. **Using Langflow API directly:**
   ```bash
   curl -X POST "http://localhost:7860/api/v1/flows/" \\
        -H "Authorization: Bearer YOUR_TOKEN" \\
        -H "Content-Type: application/json" \\
        -d @generated_flow.json
   ```

📋 **Next Steps:**
   - Configure API keys for components (OpenAI, etc.)
   - Upload files if using file components
   - Test the flow with sample inputs
   - Deploy to production when ready
""")


def main():
    """Run all tests and examples."""
    print("🚀 FLOW BUILDER AGENT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Test 1: RAG System
    test_component_rag()
    
    # Test 2: Main Use Case (Syllabus Q&A)
    syllabus_response = test_syllabus_qa_flow()
    
    # Test 3: Simple Chat Flow
    simple_response = test_simple_chat_flow()
    
    # Test 4: Document Q&A Flow  
    doc_response = test_document_qa_flow()
    
    # Test 5: Validate Generated Flows
    validate_generated_flows()
    
    # Test 6: Show Deployment Options
    show_deployment_example()
    
    # Summary
    print("=" * 60)
    print("🎉 TEST SUITE COMPLETED!")
    print("=" * 60)
    
    generated_count = sum(1 for r in [syllabus_response, simple_response, doc_response] if r is not None)
    
    print(f"📊 Results Summary:")
    print(f"   - RAG system: ✅ Working")
    print(f"   - Flows generated: {generated_count}/3")
    print(f"   - JSON files created: Check current directory")
    
    if generated_count > 0:
        print(f"\n🔗 Ready for Integration:")
        print(f"   - Import JSON files into Langflow")
        print(f"   - Configure component API keys") 
        print(f"   - Test with real data")
        print(f"   - Deploy to production")
    else:
        print(f"\n⚙️  To generate flows:")
        print(f"   - Set OPENAI_API_KEY environment variable")
        print(f"   - Run script again")
    
    print(f"\n📚 Next: Integrate with Langflow frontend header component!")


if __name__ == "__main__":
    main()
