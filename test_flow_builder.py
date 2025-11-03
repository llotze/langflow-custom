#!/usr/bin/env python3
"""
Test script for Flow Builder Agent with improved configuration.
Validates setup and tests component loading from Langflow API.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flow_builder_agent.config import Config
from flow_builder_agent.rag.component_rag import ComponentRAG
from flow_builder_agent.agent import FlowBuilderAgent


def test_configuration():
    """Test that configuration is properly set up."""
    print("=" * 80)
    print("STEP 1: TESTING CONFIGURATION")
    print("=" * 80)
    
    print(Config.display_config())
    
    if Config.validate():
        print("✅ Configuration is valid!")
        return True
    else:
        print("❌ Configuration validation failed!")
        print("\n💡 Please:")
        print("   1. Copy flow_builder_agent/.env.example to flow_builder_agent/.env")
        print("   2. Add your API key (GOOGLE_API_KEY or OPENAI_API_KEY)")
        print("   3. Run this script again")
        return False


def test_component_rag():
    """Test ComponentRAG initialization and component loading."""
    print("\n" + "=" * 80)
    print("STEP 2: TESTING COMPONENT RAG")
    print("=" * 80)
    
    try:
        print("\n🔄 Initializing ComponentRAG...")
        rag = ComponentRAG()
        
        # Check if components were loaded
        if rag.components_cache:
            total_components = sum(
                len(comps) if isinstance(comps, dict) else 0
                for comps in rag.components_cache.values()
            )
            print(f"\n✅ ComponentRAG initialized successfully!")
            print(f"   - Categories loaded: {len(rag.components_cache)}")
            print(f"   - Total components: {total_components}")
            
            # Show sample categories
            print(f"\n📦 Sample Categories:")
            for cat in list(rag.components_cache.keys())[:5]:
                comps = rag.components_cache[cat]
                if isinstance(comps, dict):
                    print(f"   - {cat}: {len(comps)} components")
        else:
            print("⚠️ No components loaded (using fallback)")
        
        # Check embeddings
        if rag.component_embeddings is not None and len(rag.component_embeddings) > 0:
            print(f"\n✅ Embeddings created:")
            print(f"   - Components: {len(rag.component_embeddings)}")
            print(f"   - Dimensions: {rag.component_embeddings.shape[1]}")
        else:
            print("\n❌ No embeddings created")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing ComponentRAG: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_search():
    """Test component search functionality."""
    print("\n" + "=" * 80)
    print("STEP 3: TESTING COMPONENT SEARCH")
    print("=" * 80)
    
    try:
        rag = ComponentRAG()
        
        test_queries = [
            "OpenAI chat model",
            "chat input from user",
            "vector database storage",
            "chat output to display messages",
        ]
        
        all_passed = True
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = rag.search_components(query, top_k=3, threshold=0.2)
            
            if results:
                print(f"   ✅ Found {len(results)} components:")
                for name, info, score in results:
                    print(f"      {score:.3f} - {name}")
            else:
                print(f"   ⚠️ No components found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Error testing search: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_initialization():
    """Test FlowBuilderAgent initialization."""
    print("\n" + "=" * 80)
    print("STEP 4: TESTING AGENT INITIALIZATION")
    print("=" * 80)
    
    try:
        print("\n🔄 Initializing FlowBuilderAgent...")
        agent = FlowBuilderAgent()
        
        print(f"✅ Agent initialized successfully!")
        print(f"   - Provider: {agent.provider}")
        print(f"   - Model: {agent.model_name}")
        print(f"   - Langflow API: {agent.langflow_api_url}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "FLOW BUILDER AGENT TEST SUITE" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {
        "Configuration": test_configuration(),
        "ComponentRAG": test_component_rag(),
        "Component Search": test_component_search(),
        "Agent Initialization": test_agent_initialization(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 80)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Flow Builder Agent is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Try creating a flow:")
        print("      python flow_builder_agent/example_usage.py")
        print("   2. Read the documentation:")
        print("      cat flow_builder_agent/README.md")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common issues:")
        print("   1. Langflow backend not running:")
        print("      Solution: Run .\\start_backend.ps1")
        print("   2. API keys not configured:")
        print("      Solution: Copy .env.example to .env and add your keys")
        print("   3. Dependencies not installed:")
        print("      Solution: pip install -r flow_builder_agent/requirements.txt")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
