"""
Quick test to verify Gemini integration works
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("=" * 50)
print("FLOW BUILDER AGENT - GEMINI TEST")
print("=" * 50)

# Check if Gemini API key is set
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("❌ GEMINI_API_KEY not set!")
    print("\nTo set it:")
    print('  PowerShell: $env:GEMINI_API_KEY = "your-key-here"')
    print("\nGet your key at: https://makersuite.google.com/app/apikey")
    exit(1)

print("✅ GEMINI_API_KEY is set")

# Try importing the agent
try:
    import sys
    from pathlib import Path
    # Add the flow_builder_agent directory to the path
    sys.path.insert(0, str(Path(__file__).parent))
    from agent import FlowBuilderAgent
    print("✅ FlowBuilderAgent imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Try initializing with Gemini
try:
    agent = FlowBuilderAgent(
        api_key=gemini_key,
        model_name="gemini-1.5-flash",  # Use flash for quick test
        provider="gemini"
    )
    print("✅ Agent initialized with Gemini")
    print(f"   Model: {agent.model_name}")
    print(f"   Provider: {agent.provider}")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    exit(1)

print("\n" + "=" * 50)
print("🎉 GEMINI SETUP SUCCESSFUL!")
print("=" * 50)
print("\nYou can now use the Flow Builder Agent with Gemini!")
print("\nNext step: Run the full example:")
print("  python -m flow_builder_agent.example_usage")
