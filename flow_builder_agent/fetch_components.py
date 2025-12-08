#!/usr/bin/env python3
"""
Simple tester: fetch component list from Langflow API and save as JSON.
- Reads settings from the project root .env via Config
- Does NOT print or require any API keys
- Windows-friendly (run: `python flow_builder_agent\fetch_components.py`)
- Option A: If API isn't available, fall back to local import of Langflow's get_type_dict
"""

import json
from pathlib import Path
import sys
import os
import requests

# Ensure project root is on sys.path so running this script directly can import the package
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # Import Config which loads the root .env
    from flow_builder_agent.config import Config
except Exception as e:
    print(f"❌ Cannot import Config: {e}")
    sys.exit(1)


def _auth_headers() -> dict:
    """Return headers with x-api-key if LANGFLOW_API_KEY is set."""
    key = os.getenv("LANGFLOW_API_KEY") or os.getenv("API_KEY")
    return {"x-api-key": key} if key else {}


def check_health(base_url: str) -> None:
    """Quick health probe (non-fatal)."""
    base = base_url.rstrip("/")
    url = f"{base}/health"
    try:
        print(f"Health check: {url}")
        r = requests.get(url, headers=_auth_headers(), timeout=5)
        print(f"   Health status: {r.status_code}")
    except Exception as e:
        print(f"   Health check failed (continuing): {e}")


def save_components(data, out_path: Path) -> None:
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved components to: {out_path}")


def fetch_via_api(base_url: str):
    base = base_url.rstrip("/")
    candidates = [
        "/api/v1/all",               # built-in/provider components (preferred)
        "/api/v1/components",        # legacy/alternative built-ins
        "/api/v1/store/components",  # community store (fallback only)
    ]
    print("=" * 80)
    print("Fetching Langflow components via API...")
    # Quick probe
    check_health(base)
    last_err = None
    headers = _auth_headers()
    if headers:
        print("Using x-api-key from environment for API calls")
    for path in candidates:
        endpoint = f"{base}{path}"
        try:
            print(f"Trying: {endpoint}")
            # Some endpoints can take a while to build the catalog on first hit
            resp = requests.get(endpoint, headers=headers, timeout=60)
            if resp.status_code == 404:
                # Try next candidate path
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, (dict, list)):
                raise ValueError("Unexpected response format (not a JSON object/array)")
            print(f"✔ Using endpoint: {endpoint}")
            return data
        except Exception as e:
            last_err = e
            print(f"   Failed: {e}")
    if last_err:
        raise last_err
    raise RuntimeError("No API endpoint returned data")


def add_langflow_to_path() -> None:
    """Ensure local Langflow packages are importable.
    Adds both repo 'src' (for 'lfx', etc.) and 'src/backend/base' (for 'langflow').
    """
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    base_dir = src_dir / "backend" / "base"
    for p in (src_dir, base_dir):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def fetch_via_local_import():
    print("=" * 80)
    print("Fetching Langflow components via local import...")
    add_langflow_to_path()
    from langflow.interface.types import get_type_dict
    return get_type_dict()


def main() -> int:
    api_base = (Config.LANGFLOW_API_URL or "http://127.0.0.1:7860").rstrip("/")
    out_path = Path(__file__).parent / "components.json"

    # Try API first
    try:
        data = fetch_via_api(api_base)
        save_components(data, out_path)
        # Small summary (handle dict or list)
        if isinstance(data, dict):
            categories = list(data.keys())
            total = sum(
                (len(v) if isinstance(v, (dict, list)) else 0) for v in data.values()
            )
            print(f"   Categories: {len(categories)} | Total items (approx): {total}")
        elif isinstance(data, list):
            print(f"   Total components: {len(data)}")
        print("=" * 80)
        return 0
    except Exception as api_err:
        print(f"API fetch failed: {api_err}")

    # Fallback to local import
    try:
        data = fetch_via_local_import()
        if not isinstance(data, (dict, list)):
            raise ValueError("get_type_dict returned unexpected data shape")
        save_components(data, out_path)
        if isinstance(data, dict):
            categories = list(data.keys())
            total = sum(
                (len(v) if isinstance(v, (dict, list)) else 0) for v in data.values()
            )
            print(f"   Categories: {len(categories)} | Total items (approx): {total}")
        elif isinstance(data, list):
            print(f"   Total components: {len(data)}")
        print("=" * 80)
        return 0
    except Exception as imp_err:
        print(f"Local import failed: {imp_err}")
        print("✗ Could not obtain components. Ensure Langflow is installed or backend is running.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
