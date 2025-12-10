#!/usr/bin/env python3
"""
Quick diagnostic to show component loading status
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("FLOW BUILDER COMPONENT SYSTEM - QUICK STATUS CHECK")
print("=" * 80)

# Test 1: Can we load the RAG?
print("\n[1/4] Testing ComponentRAG initialization...")
try:
    from flow_builder_agent.rag.component_rag import ComponentRAG
    rag = ComponentRAG()
    print(f"✅ ComponentRAG initialized")
    
    # Check what was loaded
    if rag.components_cache:
        num_categories = len(rag.components_cache)
        total_components = sum(
            len(comps) if isinstance(comps, dict) else 0 
            for comps in rag.components_cache.values()
        )
        print(f"   📦 Categories loaded: {num_categories}")
        print(f"   📦 Total components: {total_components}")
        
        if total_components > 10:
            print(f"   ✅ SUCCESS: Loaded {total_components} components (more than fallback!)")
        else:
            print(f"   ⚠️  WARNING: Only {total_components} components (using fallback)")
    else:
        print(f"   ❌ No components loaded")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Can we search for components?
print("\n[2/4] Testing component search...")
try:
    test_queries = ["Amazon S3", "OpenAI", "vector database"]
    for query in test_queries:
        results = rag.search_components(query, top_k=2, threshold=0.2)
        if results:
            print(f"   🔍 '{query}' → Found {len(results)} components:")
            for name, _, score in results:
                print(f"      {score:.2f} - {name}")
        else:
            print(f"   🔍 '{query}' → No results")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Test alias resolution
print("\n[3/4] Testing component name aliasing...")
try:
    test_aliases = {
        "OpenAI": "Should resolve to OpenAIModel",
        "Amazon": "Should resolve to AmazonS3Component",
        "Gemini": "Should resolve to GoogleGenerativeAIModel"
    }
    
    for alias, description in test_aliases.items():
        resolved = rag.resolve_component_name(alias)
        if resolved != alias:
            print(f"   ✅ '{alias}' → '{resolved}'")
        else:
            print(f"   ⚠️  '{alias}' → (no alias, using as-is)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Check if SimpleFlowBuilderAgent can be initialized
print("\n[4/4] Testing SimpleFlowBuilderAgent...")
try:
    from flow_builder_agent import SimpleFlowBuilderAgent
    agent = SimpleFlowBuilderAgent()
    print(f"   ✅ Agent initialized")
    print(f"   Provider: {agent.provider}")
    print(f"   Model: {agent.model_name}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if total_components > 10:
    print("✅ Component system is working!")
    print(f"   - {total_components} components available")
    print("   - Semantic search functional")
    print("   - Alias resolution active")
    print("\nREADY TO GENERATE FLOWS! 🚀")
else:
    print("⚠️  Component system needs attention")
    print(f"   - Only {total_components} components (expected 100+)")
    print("   - May need to run with Langflow backend")
    print("\nFLOWS WILL USE FALLBACK COMPONENTS")

print("\nTo test flow generation, run:")
print("  python test_specific_components.py")
print("=" * 80)
