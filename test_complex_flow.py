#!/usr/bin/env python3
"""
Test complex flow generation with Gemini
"""

import sys
import os
import asyncio
import json
from dotenv import load_dotenv

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Load environment variables
load_dotenv()

async def test_complex_flow():
    """Test the flow builder agent with complex prompts"""
    try:
        from flow_builder_agent import SimpleFlowBuilderAgent
        
        print("=" * 80)
        print("Testing Flow Builder with COMPLEX prompts")
        print("=" * 80)
        
        agent = SimpleFlowBuilderAgent()
        
        test_prompts = [
            "Create a RAG chatbot that uses a vector database",
            "Build a document Q&A system with memory",
            "Create a chatbot with web search capabilities",
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'=' * 80}")
            print(f"TEST {i}: {prompt}")
            print("=" * 80)
            
            flow_data = await agent.build_flow_async(prompt)
            
            if flow_data.get("data", {}).get("nodes"):
                nodes = flow_data["data"]["nodes"]
                edges = flow_data["data"].get("edges", [])
                
                print(f"\n✓ Generated {len(nodes)} nodes and {len(edges)} edges:")
                
                for node in nodes:
                    node_id = node.get("id", "unknown")
                    node_type = node.get("data", {}).get("type", "unknown")
                    display_name = node.get("data", {}).get("node", {}).get("display_name", "N/A")
                    has_template = "template" in node.get("data", {}).get("node", {})
                    print(f"  📦 {node_id}")
                    print(f"     Type: {node_type}")
                    print(f"     Display: {display_name}")
                    print(f"     Has Template: {has_template}")
                
                print(f"\n  Edges:")
                for edge in edges:
                    source = edge.get("source", "?")
                    target = edge.get("target", "?")
                    print(f"    {source} → {target}")
                
                # Save the flow for inspection
                filename = f"generated_flow_{i}.json"
                with open(filename, 'w') as f:
                    json.dump(flow_data, f, indent=2)
                print(f"\n  💾 Saved to {filename}")
            else:
                print("\n✗ ERROR: No nodes generated")
        
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complex_flow())
