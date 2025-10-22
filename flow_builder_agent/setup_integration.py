#!/usr/bin/env python3
"""
Setup script for Flow Builder Agent integration with Langflow
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install Flow Builder Agent dependencies"""
    print("📦 Installing Flow Builder Agent dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_environment():
    """Check if the environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    issues = []
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY environment variable not set")
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, current: {sys.version}")
    
    # Check if we're in the right directory
    langflow_root = Path(__file__).parent.parent
    if not (langflow_root / "src" / "backend").exists():
        issues.append("Not in Langflow root directory")
    
    if issues:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Environment configuration OK")
    return True


def create_symlink():
    """Create symlink for easy import"""
    print("🔗 Setting up Flow Builder Agent integration...")
    
    # Get paths
    flow_builder_path = Path(__file__).parent
    langflow_backend = flow_builder_path.parent / "src" / "backend" / "base"
    
    if not langflow_backend.exists():
        print(f"❌ Langflow backend not found at {langflow_backend}")
        return False
    
    # Add to Python path (for development)
    print(f"✅ Flow Builder Agent path: {flow_builder_path}")
    print(f"✅ Langflow backend path: {langflow_backend}")
    
    return True


def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        from flow_builder_agent import FlowBuilderAgent
        from flow_builder_agent.rag.component_rag import ComponentRAG
        from flow_builder_agent.schemas.flow_schema import FlowGenerationRequest
        
        print("✅ Core imports successful")
        
        # Test RAG initialization
        rag = ComponentRAG()
        print(f"✅ RAG system initialized with {len(rag.components)} components")
        
        # Test component search
        results = rag.search_components("chat", top_k=3)
        print(f"✅ Component search working ({len(results)} results)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Flow Builder Agent Setup")
    print("=" * 40)
    
    success = True
    
    # Step 1: Check environment
    success = check_environment() and success
    
    # Step 2: Install dependencies
    success = install_dependencies() and success
    
    # Step 3: Setup integration
    success = create_symlink() and success
    
    # Step 4: Run tests
    success = run_tests() and success
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Start Langflow backend: `make backend`")
        print("3. Test the Flow Builder Agent API at /api/v1/flow-builder/health")
        print("4. Run integration test: `python flow_builder_agent/test_integration.py`")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
