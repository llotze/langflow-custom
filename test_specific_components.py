#!/usr/bin/env python3
"""
Test flow generation with specific components like Amazon S3
"""

import sys
import os
import asyncio
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

async def test_specific_components():
    from flow_builder_agent import SimpleFlowBuilderAgent
    
    print("=" * 80)
    print("Testing Flow Builder with Specific Component Requests")
    print("=" * 80)
    
    agent = SimpleFlowBuilderAgent()
    
    # Test prompts that should trigger specific components
    test_cases = [
        {
            "prompt": "Create a chatbot that stores conversation history in Amazon S3",
            "expected_components": ["ChatInput", "Amazon", "S3", "ChatOutput"]
        },
        {
            "prompt": "Build a document processing flow with OpenAI",
            "expected_components": ["File", "OpenAI", "ChatOutput"]
        },
        {
            "prompt": "Create a flow with Google Gemini model",
            "expected_components": ["ChatInput", "Gemini", "Google", "ChatOutput"]
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test['prompt']}")
        print(f"Expected to find: {', '.join(test['expected_components'])}")
        print("=" * 80)
        
        # First, test what the RAG finds
        print("\n🔍 RAG Search Results:")
        relevant = agent.rag.search_components(test['prompt'], top_k=5, threshold=0.2)
        
        if relevant:
            for comp_name, comp_info, score in relevant:
                print(f"  {score:.3f} - {comp_name}")
        else:
            print("  ⚠️ No components found by RAG")
        
        # Generate the flow
        print("\n🤖 Generating flow...")
        flow_data = await agent.build_flow_async(test['prompt'])
        
        nodes = flow_data.get("data", {}).get("nodes", [])
        edges = flow_data.get("data", {}).get("edges", [])
        
        print(f"\n✓ Generated {len(nodes)} nodes, {len(edges)} edges:")
        
        node_types = []
        for node in nodes:
            node_id = node.get("id", "?")
            node_type = node.get("data", {}).get("type", "unknown")
            node_types.append(node_type)
            print(f"  📦 {node_id}: {node_type}")
        
        # Check if expected components are present
        print(f"\n🔍 Component Match Analysis:")
        for expected in test['expected_components']:
            # Check if any node type contains the expected component
            found = any(expected.lower() in node_type.lower() for node_type in node_types)
            status = "✅" if found else "❌"
            print(f"  {status} Looking for '{expected}'")
        
        print()
    
    print("=" * 80)
    print("Testing Complete!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_specific_components())
