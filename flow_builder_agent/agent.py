"""
Flow Builder Agent (wrapper)
- Thin wrapper around SimpleFlowBuilderAgent for stability
- Avoids direct dependency on Langflow internal Pydantic schemas
- Returns simple flow dicts (name, description, data) ready for API usage
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from .simple_agent import SimpleFlowBuilderAgent
from .config import Config

logger = logging.getLogger(__name__)


class FlowBuilderAgent:
    """
    Stable agent interface that delegates to SimpleFlowBuilderAgent.
    Use this class in apps and tests expecting `FlowBuilderAgent`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        langflow_api_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        # Delegate to the simple agent (which already supports Gemini RAG grounding)
        self._agent = SimpleFlowBuilderAgent(
            api_key=api_key,
            model_name=model_name or Config.MODEL_NAME,
            provider=provider or Config.PROVIDER,
            langflow_api_url=langflow_api_url or Config.LANGFLOW_API_URL,
            openai_api_key=openai_api_key,
        )
        # Convenience mirrors
        self.provider = self._agent.provider
        self.model_name = self._agent.model_name
        self.langflow_api_url = self._agent.langflow_api_url
        logger.info(
            f"✅ Initialized FlowBuilderAgent wrapper with {self.provider} ({self.model_name})"
        )

    def build_flow(self, user_request: str, flow_name: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous flow builder. Returns a simple flow dict."""
        return self._agent.build_flow(user_request, flow_name)

    async def build_flow_async(self, user_request: str, flow_name: Optional[str] = None) -> Dict[str, Any]:
        """Async flow builder. Returns a simple flow dict."""
        return await self._agent.build_flow_async(user_request, flow_name)

    # Back-compat: accept a request object shaped as {"user_request": str, ...}
    async def generate_flow_async(self, request: Any) -> Dict[str, Any]:
        user_request: str = getattr(request, "user_request", None) or getattr(
            request, "query", None
        ) or (request.get("user_request") if isinstance(request, dict) else None)
        flow_name: Optional[str] = getattr(request, "flow_name", None) or (
            request.get("flow_name") if isinstance(request, dict) else None
        )
        if not user_request:
            raise ValueError("Missing 'user_request' for flow generation")
        return await self.build_flow_async(user_request, flow_name)

    def validate_flow_json(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a simple flow dict using the simple agent's validator."""
        return self._agent.validate_flow_json(flow_data)

    def deploy_flow(self, flow: Dict[str, Any], langflow_api_url: Optional[str] = None, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy the generated flow to a Langflow instance using the public API.
        Expects a flow dict with keys: name, description, data.
        """
        import requests

        url = (langflow_api_url or self.langflow_api_url).rstrip("/")
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        try:
            resp = requests.post(f"{url}/api/v1/flows/", headers=headers, json=flow, timeout=60)
            resp.raise_for_status()
            body = resp.json()
            logger.info(f"Successfully deployed flow. Response id: {body.get('id')}")
            return {"success": True, "api_response": body}
        except requests.RequestException as e:
            logger.error(f"Failed to deploy flow: {e}")
            return {"success": False, "error": str(e)}


# CLI usage for quick manual tests
if __name__ == "__main__":
    import os

    prompt = (
        "I want a bot to answer student questions about due dates using my syllabus file; "
        "send the draft reply to my inbox for approval before sending to the student."
    )

    agent = FlowBuilderAgent(api_key=os.getenv("OPENAI_API_KEY"))
    flow = agent.build_flow(prompt)
    print(json.dumps(flow, indent=2))
