# Flow Builder Agent 🚀

An intelligent AI agent that converts natural language descriptions into deployable Langflow workflows using RAG (Retrieval-Augmented Generation) and LLMs.

## 🎯 Overview

The Flow Builder Agent is a meta-AI system that:
- **Understands** natural language workflow requirements
- **Searches** available Langflow components via API
- **Generates** complete, valid Langflow JSON flows
- **Deploys** workflows automatically via Langflow API

## 🏗️ Architecture

```
┌─────────────────────┐
│  User Request       │
│  "Create a chatbot" │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Flow Builder Agent │
│  (Gemini/OpenAI)    │
└──────────┬──────────┘
           │
           ├──────────────────────┐
           ▼                      ▼
┌─────────────────────┐  ┌──────────────────┐
│  ComponentRAG       │  │  Langflow API    │
│  (Semantic Search)  │  │  (Components)    │
└──────────┬──────────┘  └──────────────────┘
           │
           ▼
┌─────────────────────┐
│  Generated Flow     │
│  (Langflow JSON)    │
└─────────────────────┘
```

## 🔧 Setup

### 1. Prerequisites

- Python 3.9+
- Langflow backend running on `http://127.0.0.1:7860`
- API key for Gemini or OpenAI

### 2. Install Dependencies

```powershell
# From project root
cd flow_builder_agent
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file and add your API keys:

```powershell
# Copy example file
Copy-Item .env.example .env

# Edit .env and add your keys
notepad .env
```

**Required settings in `.env`:**

```bash
# Choose your LLM provider
FLOW_BUILDER_PROVIDER=gemini  # or "openai"

# Add your API key (get from provider)
GOOGLE_API_KEY=your_gemini_key_here
# OR
OPENAI_API_KEY=your_openai_key_here

# Langflow backend URL (default is correct for local dev)
LANGFLOW_API_URL=http://127.0.0.1:7860
```

**Get API Keys:**
- **Gemini**: https://makersuite.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys

### 4. Start Langflow Backend

The Flow Builder Agent requires Langflow backend to be running:

```powershell
# Option 1: Use the startup script (recommended)
.\start_backend.ps1

# Option 2: Manual start
uv run uvicorn --factory langflow.main:create_app --host 0.0.0.0 --port 7860 --reload
```

Wait until you see: `Uvicorn running on http://127.0.0.1:7860`

## 🚀 Usage

### Basic Example

```python
from flow_builder_agent.agent import FlowBuilderAgent
from flow_builder_agent.config import Config

# Display configuration (safe - masks API keys)
print(Config.display_config())

# Initialize agent (uses .env configuration)
agent = FlowBuilderAgent()

# Create a flow from natural language
request = "Create a simple chatbot using OpenAI that can answer user questions"
flow_json = agent.create_flow(request)

# Deploy the flow to Langflow
flow_id = agent.deploy_flow(flow_json)
print(f"✅ Flow deployed with ID: {flow_id}")
```

### Advanced Usage

```python
# Override config values
agent = FlowBuilderAgent(
    provider="gemini",
    model_name="gemini-2.5-flash",
    api_key="your_custom_key",
    langflow_api_url="http://custom-host:8080"
)

# Create complex flows
request = """
Create a RAG system that:
1. Loads PDF documents
2. Splits them into chunks
3. Creates embeddings using OpenAI
4. Stores in Pinecone vector database
5. Answers questions using retrieval
"""

flow = agent.create_flow(request)
```

## 🔍 How It Works

### 1. Component Discovery

The agent fetches available components from Langflow API:

```python
# ComponentRAG queries Langflow API
GET http://127.0.0.1:7860/api/v1/all

# Returns all component types organized by category:
{
  "inputs": {"ChatInput": {...}, ...},
  "models": {"OpenAIModel": {...}, "GoogleGenerativeAIModel": {...}},
  "vectorstores": {"PineconeVectorStore": {...}, ...},
  ...
}
```

### 2. Semantic Search

When you request a flow, the RAG system:
- Converts your request into embeddings
- Searches for relevant components
- Ranks by semantic similarity
- Returns top matches with metadata

### 3. Flow Generation

The LLM (Gemini/OpenAI):
- Reviews your request
- Analyzes available components
- Plans the workflow architecture
- Generates valid Langflow JSON

### 4. Validation & Deployment

- Validates JSON structure
- Checks component compatibility
- Deploys to Langflow backend
- Returns flow ID for execution

## 📁 Project Structure

```
flow_builder_agent/
├── __init__.py           # Package initialization
├── agent.py              # Main FlowBuilderAgent class
├── config.py             # Configuration management (NEW)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template (NEW)
│
├── rag/
│   ├── __init__.py
│   └── component_rag.py  # RAG system for component search
│
├── schemas/
│   ├── __init__.py
│   └── flow_schema.py    # Pydantic models for flows
│
└── knowledge_base/
    └── components.json   # Fallback component definitions
```

## 🔐 Security Best Practices

### ✅ DO:
- Store API keys in `.env` file
- Add `.env` to `.gitignore`
- Use environment variables in production
- Rotate API keys regularly

### ❌ DON'T:
- Commit API keys to git
- Hardcode keys in source code
- Share `.env` file
- Use production keys in development

### Checking Your Setup

```python
from flow_builder_agent.config import Config

# Validate configuration
if Config.validate():
    print("✅ Configuration is valid")
    print(Config.display_config())  # Shows masked keys
else:
    print("❌ Configuration has errors")
```

## 🧪 Testing

### Test Component Loading

```python
from flow_builder_agent.rag.component_rag import ComponentRAG

# Initialize RAG
rag = ComponentRAG()

# Check component loading
if rag.components_cache:
    print(f"✅ Loaded {len(rag.components_cache)} component categories")
    
    # Test search
    results = rag.search_components("OpenAI chat model", top_k=5)
    for name, info, score in results:
        print(f"  {score:.3f} - {name}")
else:
    print("❌ Failed to load components")
```

### Test Flow Generation

```python
# Run diagnostic script
python diagnose_rag.py
```

## 🐛 Troubleshooting

### Cannot Connect to Langflow API

**Error:**
```
⚠️ Cannot connect to Langflow API at http://127.0.0.1:7860
```

**Solution:**
1. Ensure Langflow backend is running:
   ```powershell
   .\start_backend.ps1
   ```
2. Check the URL in `.env` file
3. Verify firewall settings

### API Key Not Found

**Error:**
```
ValueError: GOOGLE_API_KEY not found in environment variables
```

**Solution:**
1. Copy `.env.example` to `.env`
2. Add your API key to `.env`
3. Restart your Python session

### Component Search Returns Empty

**Error:**
```
❌ RAG: No component embeddings available!
```

**Solution:**
1. Check Langflow backend is running
2. Verify API endpoint works:
   ```powershell
   curl http://127.0.0.1:7860/api/v1/all
   ```
3. Check logs for loading errors

## 📚 API Reference

### FlowBuilderAgent

```python
class FlowBuilderAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,      # LLM API key (or use .env)
        model_name: Optional[str] = None,   # Model name (or use .env)
        provider: Optional[str] = None,     # "gemini" or "openai"
        langflow_api_url: Optional[str] = None  # Langflow backend URL
    )
    
    def create_flow(self, request: str) -> Dict[str, Any]:
        """Generate a Langflow flow from natural language."""
        
    def deploy_flow(self, flow_json: Dict[str, Any]) -> str:
        """Deploy flow to Langflow and return flow ID."""
```

### ComponentRAG

```python
class ComponentRAG:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # Embedding model
        langflow_api_url: Optional[str] = None  # Langflow API URL
    )
    
    def search_components(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Tuple[str, Any, float]]:
        """Search for components matching query."""
```

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Submit a pull request

## 📝 License

See main project LICENSE file.

## 🆘 Support

- **Documentation**: See `docs/` folder
- **Issues**: GitHub Issues
- **Langflow Docs**: https://docs.langflow.org/

---

**Made with ❤️ for the Langflow community**
