# Flow Builder Agent - 404 Error Fix

## Problem
The frontend was getting a **404 error** when trying to use the Flow Builder Agent feature:
```
Failed to load resource: the server responded with a status of 404 (Not Found)
POST http://localhost:3000/api/v1/flow_builder/build
```

## Root Cause
**URL Mismatch**: The backend API router was configured with `/flow-builder` (hyphen) but the frontend was calling `/flow_builder` (underscore).

## Changes Made

### 1. Backend API Endpoint Fixed ✅
**File**: `src/backend/base/langflow/api/v1/flow_builder.py`

- Changed router prefix from `/flow-builder` to `/flow_builder`
- Fixed syntax/indentation issues in `get_flow_builder_agent()`
- Updated to prioritize Gemini API over OpenAI

```python
# Changed from:
router = APIRouter(prefix="/flow-builder", tags=["Flow Builder Agent"])

# To:
router = APIRouter(prefix="/flow_builder", tags=["Flow Builder Agent"])
```

### 2. Gemini Support Added to SimpleFlowBuilderAgent ✅
**File**: `flow_builder_agent/simple_agent.py`

- Added `import google.generativeai as genai`
- Updated `__init__()` to support both Gemini and OpenAI providers
- Modified `_generate_flow_data_async()` to route to appropriate provider
- Added Gemini API call with JSON mode

**Key Changes**:
```python
def __init__(self, api_key: str = None, model_name: str = None, 
             provider: str = "gemini", openai_api_key: str = None):
    # Supports both providers with backwards compatibility
    if provider == "gemini":
        self.model_name = model_name or "gemini-2.5-flash"
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)
    elif provider == "openai":
        self.model_name = model_name or "gpt-4o"
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
```

### 3. Backend Configuration Updated ✅
**File**: `src/backend/base/langflow/api/v1/flow_builder.py`

Updated `get_flow_builder_agent()` to:
1. Check for `GEMINI_API_KEY` first (prioritize Gemini)
2. Fall back to `OPENAI_API_KEY` if Gemini not available
3. Raise error if neither is configured

```python
if gemini_key:
    _agent = SimpleFlowBuilderAgent(
        api_key=gemini_key,
        provider="gemini",
        model_name="gemini-2.5-flash"
    )
elif openai_key:
    _agent = SimpleFlowBuilderAgent(
        openai_api_key=openai_key
    )
```

## How It Works Now

1. **Frontend** calls: `POST /api/v1/flow_builder/build`
2. **Backend** router matches: `/flow_builder` ✅
3. **Agent** initializes with Gemini API (since `GEMINI_API_KEY` is set in `.env`)
4. **Gemini** generates the flow structure in JSON format
5. **Response** returns to frontend with flow data

## Environment Variables

The backend checks these in order:
1. `GEMINI_API_KEY` → Use Gemini (gemini-2.5-flash)
2. `OPENAI_API_KEY` → Use OpenAI (gpt-4o)

Your current `.env` has:
```env
GEMINI_API_KEY=AIzaSyB4b8o-WKnOzZ8ECaRKpxTYs1rE3U_UO8s
```

So the Flow Builder will use **Google Gemini 2.5 Flash** by default! 🎉

## Testing

The backend server is running with `--reload`, so changes are automatically applied.

**To test**:
1. Open Langflow UI at `http://localhost:3000`
2. Look for the Flow Builder chat feature
3. Type a natural language request like "Create a chatbot that uses OpenAI"
4. The agent should generate a flow using Gemini API

## Next Steps

If you still see errors:
1. Check the backend terminal for Python errors
2. Verify `google-generativeai` is installed: `.\.venv\Scripts\pip.exe list | Select-String "google-generativeai"`
3. Test the Gemini API key is working: `python flow_builder_agent\test_gemini_simple.py`

## Files Modified

1. `src/backend/base/langflow/api/v1/flow_builder.py` - Fixed endpoint, added Gemini support
2. `flow_builder_agent/simple_agent.py` - Added dual provider support (Gemini + OpenAI)

**Status**: ✅ Ready to test in the browser!
