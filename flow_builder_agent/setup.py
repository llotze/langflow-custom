"""
Setup and installation script for the Flow Builder Agent.
This script sets up the environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    requirements_file = Path("flow_builder_agent/requirements.txt")
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "Installing Flow Builder Agent dependencies"
    )


def create_env_file():
    """Create a .env file template."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_content = """# Flow Builder Agent Configuration

# OpenAI API Key (required for flow generation)
OPENAI_API_KEY=your_openai_api_key_here

# Langflow API Configuration (for deployment)
LANGFLOW_API_URL=http://localhost:7860
LANGFLOW_AUTH_TOKEN=your_langflow_auth_token_here

# Optional: Custom model settings
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("   ➡️  Please edit .env and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def test_installation():
    """Test that the installation works."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        from flow_builder_agent import FlowBuilderAgent, ComponentRAG
        from flow_builder_agent.schemas.flow_schema import LangflowFlow
        print("✅ All imports successful")
        
        # Test RAG system
        rag = ComponentRAG()
        results = rag.search_components("chat input", top_k=1)
        if results:
            print("✅ RAG system working")
        else:
            print("⚠️  RAG system returned no results")
        
        print("✅ Installation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Try running: pip install -r flow_builder_agent/requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def show_next_steps():
    """Show next steps after installation."""
    print("\n🎉 FLOW BUILDER AGENT SETUP COMPLETE!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Configure your OpenAI API key in .env file")
    print("2. Test the system:")
    print("   python flow_builder_agent/example_usage.py")
    print("\n3. Integration options:")
    print("   A. Use as Python library:")
    print("      from flow_builder_agent import FlowBuilderAgent")
    print("      agent = FlowBuilderAgent(openai_api_key='your_key')")
    print("      response = agent.build_flow('your request')")
    print("\n   B. Add to Langflow frontend (custom button)")
    print("   C. Create API endpoint for web integration")
    
    print(f"\n📚 Documentation:")
    print(f"   - See flow_builder_agent/README.md for detailed usage")
    print(f"   - Check example_usage.py for code examples")
    print(f"   - Review schemas in flow_builder_agent/schemas/")
    
    print(f"\n🔧 Architecture:")
    print(f"   - RAG System: flow_builder_agent/rag/")
    print(f"   - Knowledge Base: flow_builder_agent/knowledge_base/")
    print(f"   - Pydantic Schemas: flow_builder_agent/schemas/")
    print(f"   - Main Agent: flow_builder_agent/agent.py")


def main():
    """Main setup function."""
    print("🚀 FLOW BUILDER AGENT SETUP")
    print("=" * 40)
    
    success_steps = 0
    total_steps = 5
    
    # Step 1: Check Python version
    if check_python_version():
        success_steps += 1
    
    # Step 2: Install dependencies
    if install_dependencies():
        success_steps += 1
    
    # Step 3: Create env file
    if create_env_file():
        success_steps += 1
    
    # Step 4: Test installation
    if test_installation():
        success_steps += 1
    
    # Step 5: Show next steps
    show_next_steps()
    success_steps += 1
    
    print(f"\n📊 Setup Results: {success_steps}/{total_steps} steps completed")
    
    if success_steps == total_steps:
        print("✅ Setup completed successfully!")
        return True
    else:
        print("⚠️  Some setup steps failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    main()
