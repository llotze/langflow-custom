#!/usr/bin/env python3
"""
Quick test runner to validate Gemini-grounded RAG integration end to end.
- Ensures backend URL from .env is used
- Builds grounded context and generates a simple flow
- Prints debug artifact locations
"""
import os
import json
from pathlib import Path

from flow_builder_agent import SimpleFlowBuilderAgent
from flow_builder_agent.config import Config


def main():
    print("=" * 80)
    print("GEMINI-GROUNDED FLOW BUILDER TEST")
    print("=" * 80)

    # Show config (safe)
    print(Config.display_config())

    # Check preconditions
    if not Config.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY missing in .env")
        return 1

    # Build agent and run a sample
    agent = SimpleFlowBuilderAgent(provider="gemini", model_name=Config.MODEL_NAME)
    goal = "Create a simple chat flow using OpenAI model with ChatInput and ChatOutput"
    print(f"User goal: {goal}")
    flow = agent.build_flow(goal, flow_name="Grounded Chat Flow")

    out = Path("generated_grounded_flow.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(flow, f, indent=2)

    print(f"✅ Saved generated flow: {out}")

    kb_dir = Path("flow_builder_agent/knowledge_base")
    print("\nArtifacts:")
    print(f"  - {kb_dir / 'grounded_context.jsonl'}")
    print(f"  - {kb_dir / 'last_llm_flow.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
