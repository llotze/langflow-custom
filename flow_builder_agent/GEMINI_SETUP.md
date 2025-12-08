# Using Gemini with Flow Builder Agent

The Flow Builder Agent now supports both **Google Gemini** and **OpenAI** models!

## Setup

### 1. Install Dependencies

```powershell
cd D:\langflow_spark\langflow-AI
pip install google-generativeai
```

Or install all flow_builder_agent requirements:

```powershell
pip install -r flow_builder_agent/requirements.txt
```

### 2. Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### 3. Set Environment Variable

**In PowerShell:**
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
```

## Usage

### Option 1: Use Gemini (Default)

```python
from flow_builder_agent import FlowBuilderAgent

# Initialize with Gemini
agent = FlowBuilderAgent(
    api_key="your-gemini-api-key",
    model_name="gemini-1.5-pro",  # or "gemini-1.5-flash" for faster/cheaper
    provider="gemini"
)

# Build a flow
result = agent.build_flow("Create a chatbot that answers questions about my documents")
print(result.flow.dict())
```

### Option 2: Use OpenAI

```python
agent = FlowBuilderAgent(
    api_key="your-openai-api-key",
    model_name="gpt-4o",
    provider="openai"
)
```

### Option 3: Backwards Compatible (OpenAI)

```python
# Old way still works
agent = FlowBuilderAgent(openai_api_key="your-openai-key")
```

## Running the Example

```powershell
# Set your API key
$env:GEMINI_API_KEY = "your-key-here"

# Run the example
cd D:\langflow_spark\langflow-AI
python -m flow_builder_agent.example_usage
```

The example will automatically use Gemini if `GEMINI_API_KEY` is set, otherwise it falls back to OpenAI.

## Model Options

### Gemini Models
- `gemini-1.5-pro` - Most capable, best for complex flows
- `gemini-1.5-flash` - Faster and cheaper, good for simpler flows
- `gemini-1.0-pro` - Previous generation

### OpenAI Models
- `gpt-4o` - Most capable (default)
- `gpt-4-turbo` - Fast and capable
- `gpt-3.5-turbo` - Faster and cheaper

## Benefits of Using Gemini

✅ **Free tier** - 60 requests/minute free quota  
✅ **Large context window** - Up to 2M tokens  
✅ **Cost-effective** - Generally cheaper than OpenAI  
✅ **JSON mode** - Native structured output support  

## Troubleshooting

If you get import errors:
```powershell
pip install --upgrade google-generativeai
```

If Gemini API fails:
```powershell
# Check your API key is set
echo $env:GEMINI_API_KEY

# Try using OpenAI as fallback
$env:OPENAI_API_KEY = "your-openai-key"
```
