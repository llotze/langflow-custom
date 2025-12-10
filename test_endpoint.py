#!/usr/bin/env python3
"""
Test the Flow Builder HTTP endpoint directly
"""

import requests
import json

def test_flow_builder_endpoint():
    """Test the /api/v1/flow_builder/build endpoint"""
    
    url = "http://127.0.0.1:7860/api/v1/flow_builder/build"
    
    test_prompts = [
        "Create a simple chatbot",
        "Build a RAG system with vector database",
        "Create a chatbot with web search"
    ]
    
    print("=" * 80)
    print("Testing Flow Builder HTTP Endpoint")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {prompt}")
        print("=" * 80)
        
        try:
            response = requests.post(
                url,
                json={"user_request": prompt},
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                flow_data = response.json()
                nodes = flow_data.get("data", {}).get("nodes", [])
                edges = flow_data.get("data", {}).get("edges", [])
                
                print(f"✓ Generated {len(nodes)} nodes, {len(edges)} edges")
                
                # Check template presence
                templates_ok = True
                for node in nodes:
                    node_id = node.get("id")
                    node_type = node.get("data", {}).get("type")
                    has_template = "template" in node.get("data", {}).get("node", {})
                    
                    if not has_template:
                        print(f"  ⚠️ {node_id} ({node_type}) - Missing template!")
                        templates_ok = False
                    else:
                        print(f"  ✓ {node_id} ({node_type}) - Has template")
                
                if templates_ok:
                    print(f"\n✅ All nodes have proper templates!")
                else:
                    print(f"\n⚠️ Some nodes missing templates")
                
                # Save flow
                filename = f"endpoint_flow_{i}.json"
                with open(filename, 'w') as f:
                    json.dump(flow_data, f, indent=2)
                print(f"💾 Saved to {filename}")
                
            else:
                print(f"❌ Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to backend. Is it running?")
            print("   Start with: make backend")
            break
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Endpoint testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_flow_builder_endpoint()
