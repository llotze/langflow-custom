"""
Gemini-backed RAG over Langflow built-in components.
- Loads catalog from flow_builder_agent/components.json (or via API URL in Config if provided)
- Builds embeddings with Google text-embedding-004
- Retrieves top-K relevant components and crafts a grounded prompt for Gemini

Minimal deps: google-generativeai, numpy, requests
Index persisted under flow_builder_agent/knowledge_base/
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import requests
import logging

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # Will error at runtime if used without being installed

from ..config import Config

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)
EMB_FILE = KB_DIR / "components_embed.npy"
META_FILE = KB_DIR / "components_meta.json"
CATALOG_FILE = Path(__file__).resolve().parents[1] / "components.json"

EMBED_MODEL = "text-embedding-004"
DEFAULT_CHAT_MODEL = Config.MODEL_NAME or "gemini-1.5-pro"

# Provider-focused default filters (kept small for efficiency)
DEFAULT_PROVIDER_TOKENS = [
    "openai",
    "azure",
    "google",
    "gemini",
    "vertex",
    "anthropic",
    "bedrock",
    "amazon",
    "aws",
    "mistral",
    "groq",
    "cohere",
    "ollama",
    "hugging",
    "openrouter",
    "together",
]


class GeminiComponentRAG:
    def __init__(self, langflow_api_url: Optional[str] = None, *, category_tokens: Optional[list[str]] = None):
        self.langflow_api_url = (langflow_api_url or Config.LANGFLOW_API_URL).rstrip("/")
        self.catalog: Dict[str, Any] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.meta: List[Dict[str, Any]] = []
        # Lowercased tokens used to include categories; None means include all
        self.category_tokens = [t.lower() for t in (category_tokens or DEFAULT_PROVIDER_TOKENS)]
        self._ensure_gemini()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"GeminiComponentRAG init: api_url={self.langflow_api_url}, tokens={self.category_tokens}"
        )

    def _ensure_gemini(self) -> None:
        if genai is None:
            raise RuntimeError("google-generativeai not installed. Add it to your environment.")
        if not Config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing in .env.")
        genai.configure(api_key=Config.GEMINI_API_KEY)

    # ---------------------------- Catalog loading ----------------------------
    def load_catalog(self) -> Dict[str, Any]:
        if CATALOG_FILE.exists():
            self.logger.info(f"Loading components catalog from file: {CATALOG_FILE}")
            self.catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
            return self.catalog
        # Fallback: fetch from API if file not present
        url_candidates = [
            f"{self.langflow_api_url}/api/v1/all",
            f"{self.langflow_api_url}/api/v1/components",
        ]
        last_err = None
        for url in url_candidates:
            try:
                self.logger.info(f"Fetching components catalog via API: {url}")
                r = requests.get(url, timeout=60)
                if r.status_code == 404:
                    self.logger.info(f"Endpoint not found (404), trying next: {url}")
                    continue
                r.raise_for_status()
                self.catalog = r.json()
                # Persist for reuse
                CATALOG_FILE.write_text(json.dumps(self.catalog, indent=2), encoding="utf-8")
                self.logger.info(f"Saved fetched catalog to {CATALOG_FILE}")
                return self.catalog
            except Exception as e:
                last_err = e
                self.logger.warning(f"Catalog fetch failed at {url}: {e}")
        raise RuntimeError(f"Could not load catalog from file or API: {last_err}")

    # -------------------------- Flatten + index build ------------------------
    @staticmethod
    def _flatten_catalog(catalog: Dict[str, Any], *, category_tokens: Optional[list[str]] = None) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for category, comps in catalog.items():
            if not isinstance(comps, dict):
                continue
            cat_l = str(category).lower()
            if category_tokens:
                # Include only categories that match any token
                if not any(tok in cat_l for tok in category_tokens):
                    continue
            for name, info in comps.items():
                if not isinstance(info, dict):
                    continue
                doc = {
                    "category": category,
                    "name": name,
                    "display_name": info.get("display_name", name),
                    "description": info.get("description", ""),
                    "base_classes": info.get("base_classes", []),
                    "documentation": info.get("documentation", ""),
                    "inputs": [
                        {
                            "name": k,
                            "display_name": v.get("display_name", k),
                            "type": v.get("type", v.get("_input_type", "")),
                            "info": v.get("info", ""),
                            "required": v.get("required", False),
                        }
                        for k, v in (info.get("template", {}) or {}).items()
                        if isinstance(v, dict) and k not in {"_type", "code"}
                    ],
                }
                docs.append(doc)
        return docs

    @staticmethod
    def _doc_text(doc: Dict[str, Any]) -> str:
        parts = [
            f"Category: {doc['category']}",
            f"Name: {doc['name']}",
            f"Display: {doc.get('display_name','')}",
            f"Description: {doc.get('description','')}",
        ]
        if doc.get("base_classes"):
            parts.append("Base: " + ", ".join(doc["base_classes"]))
        if doc.get("inputs"):
            inps = []
            for i in doc["inputs"]:
                inps.append(
                    f"{i['display_name']}({i['name']}:{i.get('type','')} req={i.get('required',False)}): {i.get('info','')}"
                )
            parts.append("Inputs: " + "; ".join(inps)[:800])
        if doc.get("documentation"):
            parts.append(f"Docs: {doc['documentation']}")
        return " \n".join(parts)

    def build_index(self, force: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if EMB_FILE.exists() and META_FILE.exists() and not force:
            self.logger.info(f"Loading existing embeddings: {EMB_FILE}, meta: {META_FILE}")
            self.embeddings = np.load(EMB_FILE)
            self.meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            return self.embeddings, self.meta

        catalog = self.catalog or self.load_catalog()
        docs = self._flatten_catalog(catalog, category_tokens=self.category_tokens)
        self.logger.info(f"Building embeddings for {len(docs)} provider-focused components...")
        texts = [self._doc_text(d) for d in docs]

        vectors: List[List[float]] = []
        for idx, t in enumerate(texts):
            # Gemini embedding call
            res = genai.embed_content(model=EMBED_MODEL, content=t)
            vec = res.get("embedding") or res.get("data", {}).get("embedding")
            if vec is None:
                # Persist the bad case for debugging
                (KB_DIR / f"embed_error_{idx}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
                raise RuntimeError("Embedding response missing 'embedding'")
            vectors.append(vec)

        self.embeddings = np.asarray(vectors, dtype=np.float32)
        self.meta = docs
        np.save(EMB_FILE, self.embeddings)
        META_FILE.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        self.logger.info(
            f"Saved embeddings to {EMB_FILE} (shape={self.embeddings.shape}) and meta to {META_FILE}"
        )
        return self.embeddings, self.meta

    # ------------------------------- Retrieval -------------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    def retrieve(self, query: str, top_k: int = 10, threshold: float = 0.25) -> List[Tuple[Dict[str, Any], float]]:
        if self.embeddings is None or not self.meta:
            self.build_index()
        self.logger.info(f"Retrieving top-{top_k} for query='{query[:80]}...' (threshold={threshold})")
        res = genai.embed_content(model=EMBED_MODEL, content=query)
        q = np.asarray(res.get("embedding"), dtype=np.float32)
        sims = [self._cosine_sim(q, v) for v in self.embeddings]  # type: ignore[arg-type]
        ranked = sorted(zip(self.meta, sims), key=lambda x: x[1], reverse=True)
        top = [r for r in ranked[:top_k] if r[1] >= threshold]
        self.logger.info(
            "Top matches: "
            + ", ".join([f"{d['name']}@{d['category']}({score:.2f})" for d, score in top[:5]])
        )
        return top

    # --------------------------- Prompt construction -------------------------
    @staticmethod
    def _format_context(items: List[Dict[str, Any]]) -> str:
        lines = []
        for d in items:
            lines.append(
                json.dumps(
                    {
                        "category": d["category"],
                        "name": d["name"],
                        "display_name": d.get("display_name"),
                        "description": d.get("description"),
                        "base_classes": d.get("base_classes", []),
                        "documentation": d.get("documentation"),
                        "inputs": d.get("inputs", []),
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines)

    def ask_gemini(self, user_goal: str, top_k: int = 8) -> str:
        matches = self.retrieve(user_goal, top_k=top_k)
        context_items = [m[0] for m in matches]
        context = self._format_context(context_items)

        system = (
            "You are an assistant that designs Langflow flows. "
            "Use ONLY the components listed in CONTEXT. If a needed component is missing, say 'Insufficient components'. "
            "Respond with strict JSON using keys: nodes, edges. "
            "Each node must reference a component by its 'name' exactly."
        )
        user = (
            f"USER_GOAL:\n{user_goal}\n\nCONTEXT (JSON lines of allowed components):\n{context}\n\n"
            "Return JSON only."
        )

        model_name = DEFAULT_CHAT_MODEL
        model = genai.GenerativeModel(model_name)
        self.logger.info(f"Calling Gemini model={model_name} with {len(context_items)} context items...")
        resp = model.generate_content([system, user])
        text = resp.text or ""
        self.logger.info(f"Gemini response length: {len(text)} chars")
        # Persist last response
        (KB_DIR / "last_gemini_response.json").write_text(text, encoding="utf-8")
        return text


def quick_demo(query: str) -> str:
    rag = GeminiComponentRAG()
    rag.load_catalog()
    rag.build_index()
    return rag.ask_gemini(query)
