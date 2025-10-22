"""
Test script for the Flow Builder Agent integration.
"""

import os
import json
import asyncio
from datetime import datetime


def test_simple_agent():
    """Test the simplified Flow Builder Agent."""
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        print("   For testing without OpenAI, the agent will use fallback flows.")
        
        # Test without API key (will use fallback)
        from flow_builder_agent.simple_agent import SimpleFlowBuilderAgent
        agent = SimpleFlowBuilderAgent(openai_api_key="dummy_key")
    else:
        print("✅ OpenAI API key found. Testing with real LLM integration.")
        from flow_builder_agent.simple_agent import SimpleFlowBuilderAgent
        agent = SimpleFlowBuilderAgent(openai_api_key=api_key)
    
    # Test cases
    test_cases = [
        "Create a simple chat bot",
        "Build a document Q&A system using uploaded PDF files",
        "Make a flow that sends email notifications when processing is complete",
        "Create a syllabus Q&A responder for students"
    ]
    
    print("\n🚀 Testing Flow Builder Agent...")
    print("=" * 50)
    
    for i, request in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {request}")
        print("-" * 40)
        
        try:
            # Generate flow
            start_time = datetime.now()
            flow = agent.build_flow(request)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Flow generated successfully in {processing_time:.2f}s")
            print(f"   Name: {flow['name']}")
            print(f"   Description: {flow['description']}")
            print(f"   Nodes: {len(flow['data']['nodes'])}")
            print(f"   Edges: {len(flow['data']['edges'])}")
            
            # Validate the flow
            validation = agent.validate_flow_json(flow['data'])
            if validation['valid']:
                print("   ✅ Flow validation passed")
            else:
                print("   ❌ Flow validation failed:")
                for error in validation['errors']:
                    print(f"      - {error}")
            
            if validation['warnings']:
                print("   ⚠️  Warnings:")
                for warning in validation['warnings']:
                    print(f"      - {warning}")
            
            # Save flow for inspection
            filename = f"test_flow_{i}.json"
            with open(filename, 'w') as f:
                json.dump(flow, f, indent=2)
            print(f"   💾 Flow saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error generating flow: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 Flow Builder Agent testing completed!")


def test_rag_system():
    """Test the RAG component search system."""
    
    print("\n🔍 Testing RAG Component Search...")
    print("=" * 50)
    
    try:
        from flow_builder_agent.rag.component_rag import ComponentRAG
        rag = ComponentRAG()
        
        # Test component searches
        search_queries = [
            "chat with users",
            "process files and documents", 
            "send emails",
            "language model AI",
            "vector database search"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching: '{query}'")
            results = rag.search_components(query, top_k=3)
            
            if results:
                print(f"   Found {len(results)} components:")
                for name, component, score in results:
                    print(f"   - {name}: {score:.3f}")
            else:
                print("   No components found")
        
        print("\n✅ RAG system test completed")
        
    except Exception as e:
        print(f"❌ RAG system test failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_async_functionality():
    """Test async functionality of the agent."""
    
    print("\n⚡ Testing Async Functionality...")
    print("=" * 50)
    
    try:
        api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
        from flow_builder_agent.simple_agent import SimpleFlowBuilderAgent
        agent = SimpleFlowBuilderAgent(openai_api_key=api_key)
        
        # Test async flow generation
        request = "Create a basic chatbot flow"
        
        start_time = datetime.now()
        flow = await agent.build_flow_async(request)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"✅ Async flow generated in {processing_time:.2f}s")
        print(f"   Name: {flow['name']}")
        print(f"   Nodes: {len(flow['data']['nodes'])}")
        
        # Save async test result
        with open("async_test_flow.json", 'w') as f:
            json.dump(flow, f, indent=2)
        print("   💾 Async flow saved to async_test_flow.json")
        
    except Exception as e:
        print(f"❌ Async test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 Flow Builder Agent - Integration Tests")
    print("=" * 60)
    
    # Test 1: Simple Agent
    test_simple_agent()
    
    # Test 2: RAG System
    test_rag_system()
    
    # Test 3: Async Functionality
    print("\n⚡ Running async test...")
    asyncio.run(test_async_functionality())
    
    print("\n" + "=" * 60)
    print("🎊 All tests completed!")
    print("\nNext steps:")
    print("1. Review generated flow JSON files")
    print("2. Test API endpoints with a running Langflow server")
    print("3. Import generated flows into Langflow UI")


if __name__ == "__main__":
    main()
