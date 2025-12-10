# Flow Builder Agent - Quick Start Guide 🚀

This guide will help you set up and test the Flow Builder Agent in under 5 minutes.

## Prerequisites ✅

- Python 3.9+ installed
- Langflow project (already in `d:\langflow_spark\langflow-AI`)
- Internet connection (for downloading model embeddings)

## Step-by-Step Setup

### 1️⃣ Install Dependencies

Open PowerShell and run:

```powershell
cd d:\langflow_spark\langflow-AI\flow_builder_agent
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` (for semantic search)
- `google-generativeai` (for Gemini)
- `openai` (optional, for GPT models)
- `requests` (for API calls)
- Other dependencies

### 2️⃣ Get Your API Key

**For Google Gemini (Recommended - Free tier available):**
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)

**For OpenAI (Alternative):**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)

### 3️⃣ Configure Environment

```powershell
# Copy the example environment file
cd d:\langflow_spark\langflow-AI\flow_builder_agent
Copy-Item .env.example .env

# Edit the .env file
notepad .env
```

In the `.env` file, set:

```bash
# Choose your provider
FLOW_BUILDER_PROVIDER=gemini  # or "openai"

# Add your API key (choose ONE)
GOOGLE_API_KEY=AIza_your_key_here
# OR
OPENAI_API_KEY=sk-your_key_here

# Langflow API URL (default is fine for local dev)
LANGFLOW_API_URL=http://127.0.0.1:7860
```

**Save and close the file.**

### 4️⃣ Start Langflow Backend

The Flow Builder Agent needs Langflow running to fetch component information.

**Option A: Use the PowerShell script (easiest)**
```powershell
cd d:\langflow_spark\langflow-AI
.\start_backend.ps1
```

**Option B: Manual start**
```powershell
cd d:\langflow_spark\langflow-AI
uv run uvicorn --factory langflow.main:create_app --host 0.0.0.0 --port 7860 --reload
```

**Wait for:**
```
INFO:     Uvicorn running on http://127.0.0.1:7860 (Press CTRL+C to quit)
```

### 5️⃣ Test the Setup

Open a **new PowerShell window** (keep the backend running in the first one):

```powershell
cd d:\langflow_spark\langflow-AI
python test_flow_builder.py
```

Expected output:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FLOW BUILDER AGENT TEST SUITE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1: TESTING CONFIGURATION
================================================================================
✅ Configuration is valid!

STEP 2: TESTING COMPONENT RAG
================================================================================
✅ ComponentRAG initialized successfully!
   - Categories loaded: 20
   - Total components: 150+

STEP 3: TESTING COMPONENT SEARCH
================================================================================
🔍 Query: 'OpenAI chat model'
   ✅ Found 3 components:
      0.856 - OpenAIModel
      0.721 - ChatOpenAI
      0.654 - OpenAIEmbeddings

...

TEST SUMMARY
================================================================================
Configuration.................................... ✅ PASSED
ComponentRAG..................................... ✅ PASSED
Component Search................................. ✅ PASSED
Agent Initialization............................. ✅ PASSED

🎉 All tests passed! Flow Builder Agent is ready to use.
```

## Troubleshooting 🔧

### Problem: "Cannot connect to Langflow API"

**Solution:**
1. Make sure Langflow backend is running in another terminal
2. Check that it's running on port 7860
3. Try accessing http://127.0.0.1:7860/api/v1/all in your browser

### Problem: "API key not found"

**Solution:**
1. Make sure you created the `.env` file (not `.env.example`)
2. Check that you added your API key
3. Restart your Python session/terminal after editing `.env`

### Problem: "The token '&&' is not a valid statement separator"

**Solution:**
This is a PowerShell syntax error. Instead of:
```powershell
# ❌ DON'T DO THIS
cd d:\langflow_spark\langflow-AI && uv run uvicorn ...
```

Use separate commands:
```powershell
# ✅ DO THIS
cd d:\langflow_spark\langflow-AI
uv run uvicorn --factory langflow.main:create_app --host 0.0.0.0 --port 7860
```

Or use the script:
```powershell
.\start_backend.ps1
```

### Problem: "Module not found"

**Solution:**
Install dependencies:
```powershell
cd d:\langflow_spark\langflow-AI\flow_builder_agent
pip install -r requirements.txt
```

## Next Steps 🎯

Once all tests pass, you can:

### 1. Create Your First Flow

```python
from flow_builder_agent.simple_agent import SimpleFlowBuilderAgent

# Initialize the agent (uses your .env config)
agent = SimpleFlowBuilderAgent()

# Create a flow
flow = await agent.build_flow_async(
    "Create a chatbot using OpenAI that answers user questions"
)

print(f"✅ Created flow: {flow['name']}")
print(f"   Nodes: {len(flow['data']['nodes'])}")
```

### 2. Try Different Flows

```python
# RAG system
flow = await agent.build_flow_async(
    "Build a RAG system that reads PDFs and answers questions"
)

# Data processing
flow = await agent.build_flow_async(
    "Load CSV files, process data, and store in a database"
)

# Web scraper
flow = await agent.build_flow_async(
    "Scrape websites and extract structured data"
)
```

### 3. Explore the Documentation

- **Full README**: `flow_builder_agent/README.md`
- **API Reference**: See README for detailed API docs
- **Examples**: `flow_builder_agent/example_usage.py`

## Architecture Overview 🏗️

```
Your Request
     ↓
┌─────────────────────┐
│ FlowBuilderAgent    │  ← Uses Gemini/OpenAI to understand
│ (LLM)               │    your request
└──────────┬──────────┘
           │
           ├──────────┐
           ↓          ↓
┌──────────────┐  ┌────────────────┐
│ ComponentRAG │  │ Langflow API   │  ← Fetches available
│ (Search)     │  │ /api/v1/all    │    components
└──────────────┘  └────────────────┘
           ↓
┌─────────────────────┐
│ Generated Flow JSON │  ← Ready to deploy!
└─────────────────────┘
```

## Security Notes 🔐

- ✅ API keys stored in `.env` (never committed to git)
- ✅ `.env` is in `.gitignore` automatically
- ✅ Config module masks keys in logs
- ❌ Never hardcode API keys in source code
- ❌ Never share your `.env` file

## Getting Help 🆘

1. **Check the logs**: The agent logs detailed information
2. **Run diagnostics**: `python diagnose_rag.py`
3. **Read the docs**: `flow_builder_agent/README.md`
4. **Check issues**: GitHub Issues page

## Summary Checklist ✓

Before using the agent, make sure:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] API key configured in `.env` file
- [ ] Langflow backend running on port 7860
- [ ] All tests pass (`python test_flow_builder.py`)

**You're all set! Start building flows with natural language! 🎉**

---

*For more details, see the full README at `flow_builder_agent/README.md`*
