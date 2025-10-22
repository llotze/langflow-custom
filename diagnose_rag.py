#!/usr/bin/env python3
"""
Diagnostic script to check what components are actually loaded
"""

import sys
import os
import asyncio

# Add project to path
sys.path.insert(0, r'd:\langflow_spark\langflow-AI')

from flow_builder_agent.rag.component_rag import ComponentRAG

async def diagnose_components():
    """Check what components are loaded in the RAG"""
    
    print("=" * 80)
    print("COMPONENT RAG DIAGNOSTIC")
    print("=" * 80)
    
    # Initialize RAG
    print("\nInitializing ComponentRAG...")
    rag = ComponentRAG()
    
    print(f"\n✓ RAG initialized")
    print(f"  - Langflow available: {rag.components_cache is not None}")
    print(f"  - Number of categories: {len(rag.components_cache) if rag.components_cache else 0}")
    
    if rag.components_cache:
        print(f"\n📦 LOADED COMPONENT CATEGORIES:")
        for category, components in rag.components_cache.items():
            if isinstance(components, dict):
                print(f"\n  {category.upper()} ({len(components)} components):")
                for comp_name in list(components.keys())[:5]:  # Show first 5
                    comp_info = components[comp_name]
                    if isinstance(comp_info, dict):
                        desc = comp_info.get('description', 'No description')[:60]
                        print(f"    - {comp_name}: {desc}")
                if len(components) > 5:
                    print(f"    ... and {len(components) - 5} more")
    
    # Check embeddings
    print(f"\n🔢 EMBEDDINGS:")
    if rag.component_embeddings is not None:
        print(f"  - Created: YES")
        print(f"  - Number of components: {len(rag.component_embeddings)}")
        print(f"  - Embedding dimension: {rag.component_embeddings.shape[1] if len(rag.component_embeddings) > 0 else 0}")
    else:
        print(f"  - Created: NO")
    
    # Test searches
    print(f"\n{'=' * 80}")
    print("TESTING RAG SEARCHES")
    print("=" * 80)
    
    test_queries = [
        "chat input for user messages",
        "OpenAI language model",
        "Amazon S3 file storage",
        "vector database",
        "web search",
        "chat output to display messages"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = rag.search_components(query, top_k=3, threshold=0.2)
        
        if results:
            print(f"  Found {len(results)} components:")
            for name, info, score in results:
                print(f"    {score:.3f} - {name}")
        else:
            print(f"  ❌ No components found!")
    
    # Try to reload from Langflow
    print(f"\n{'=' * 80}")
    print("ATTEMPTING TO RELOAD FROM LANGFLOW API")
    print("=" * 80)
    
    try:
        from langflow.interface.types import get_type_dict
        print("\n✓ Found Langflow's get_type_dict function")
        
        all_types = get_type_dict()
        print(f"\n📦 LANGFLOW COMPONENT TYPES:")
        print(f"  Total categories: {len(all_types)}")
        
        for category in list(all_types.keys())[:10]:
            components = all_types[category]
            print(f"\n  {category} ({len(components)} components):")
            for comp_name in list(components.keys())[:3]:
                comp_data = components[comp_name]
                display_name = comp_data.get('display_name', comp_name)
                print(f"    - {comp_name} ({display_name})")
            if len(components) > 3:
                print(f"    ... and {len(components) - 3} more")
        
        if len(all_types) > 10:
            print(f"\n  ... and {len(all_types) - 10} more categories")
        
    except ImportError as e:
        print(f"\n❌ Could not import get_type_dict: {e}")
        print("   This means we need to use a different approach to load components")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(diagnose_components())
