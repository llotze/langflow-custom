# Test Flow Builder Agent on Windows
# This script tests the Flow Builder Agent to see why LLM generation is failing

Write-Host "=== Flow Builder Agent Test (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# Set up environment
$env:PYTHONPATH = "d:\langflow_spark\langflow-AI"
$venvPath = "d:\langflow_spark\langflow-AI\.venv\Scripts\python.exe"

# Check if virtual environment exists
if (-Not (Test-Path $venvPath)) {
    Write-Host "ERROR: Virtual environment not found at $venvPath" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $venvPath" -ForegroundColor Green
Write-Host ""

# Create test Python script
$testScript = @"
import os
import sys
import asyncio
import logging

# Add project to path
sys.path.insert(0, r'd:\langflow_spark\langflow-AI')

# Set up logging to see ALL checkpoints and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from flow_builder_agent.simple_agent import SimpleFlowBuilderAgent

async def test_flow_builder():
    print("\n" + "="*80)
    print("TESTING FLOW BUILDER AGENT")
    print("="*80 + "\n")
    
    # Get API key from environment
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set it with: `$env:GEMINI_API_KEY='your-key-here'")
        return
    
    print(f"✓ Gemini API Key found: {gemini_api_key[:10]}...")
    print("")
    
    # Initialize agent with Gemini
    print("Initializing Flow Builder Agent with Gemini...")
    agent = SimpleFlowBuilderAgent(
        api_key=gemini_api_key,
        provider="gemini",
        model_name="gemini-2.0-flash-exp"
    )
    print(f"✓ Agent initialized with provider: {agent.provider}, model: {agent.model_name}")
    print("")
    
    # Test request
    test_requests = [
        "Create a simple chatbot",
        "Build a RAG system that uses PDF files",
        "Make a sentiment analysis flow"
    ]
    
    for i, user_request in enumerate(test_requests, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {user_request}")
        print(f"{'='*80}\n")
        
        try:
            # Build flow
            flow = await agent.build_flow_async(user_request)
            
            # Display results
            print(f"\n{'─'*80}")
            print("RESULT:")
            print(f"{'─'*80}")
            print(f"Name: {flow.get('name')}")
            print(f"Description: {flow.get('description')}")
            
            nodes = flow.get('data', {}).get('nodes', [])
            print(f"\nNodes ({len(nodes)}):")
            for node in nodes:
                node_type = node.get('data', {}).get('type', 'Unknown')
                node_id = node.get('id', 'Unknown')
                print(f"  - {node_type} ({node_id})")
            
            edges = flow.get('data', {}).get('edges', [])
            print(f"\nEdges ({len(edges)}):")
            for edge in edges:
                source = edge.get('source', 'Unknown')
                target = edge.get('target', 'Unknown')
                print(f"  - {source} → {target}")
            
            # Check if this is the fallback flow (always same 3 components)
            if len(nodes) == 3:
                node_types = [n.get('data', {}).get('type') for n in nodes]
                if 'ChatInput' in node_types and 'ChatOutput' in node_types:
                    print("\n⚠️  WARNING: This looks like the FALLBACK flow!")
                    print("    LLM generation likely failed and returned hardcoded flow.")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("")
        
        # Only test first one for now
        if i == 1:
            break

# Run the test
if __name__ == "__main__":
    asyncio.run(test_flow_builder())
"@

# Write test script to temp file
$tempScript = "d:\langflow_spark\langflow-AI\temp_test_flow.py"
$testScript | Out-File -FilePath $tempScript -Encoding utf8

Write-Host "Running test script..." -ForegroundColor Yellow
Write-Host ""

# Run the test
& $venvPath $tempScript

# Clean up
Remove-Item $tempScript -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
