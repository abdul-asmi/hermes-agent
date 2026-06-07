"""Gemini Embedding 2 + PostgreSQL pgvector memory for Hermes."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_TOP_K = 5
TABLE_NAME = "hermes.memory"


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_config() -> dict:
    config = {
        "database_url": os.environ.get("DATABASE_URL", "").strip(),
        "api_key": (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        ),
        "base_url": os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL).strip(),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "top_k": DEFAULT_TOP_K,
        "auto_save": True,
        "request_timeout": 30,
        "max_memory_chars": 12000,
    }

    try:
        import yaml
        from hermes_constants import get_hermes_home

        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with config_path.open(encoding="utf-8-sig") as handle:
                root = yaml.safe_load(handle) or {}
            memory_config = root.get("memory", {})
            plugin_config = memory_config.get("gemini_pgvector", {})
            if isinstance(plugin_config, dict):
                config.update(
                    {
                        key: value
                        for key, value in plugin_config.items()
                        if value is not None and value != ""
                    }
                )
    except Exception:
        logger.debug("Could not load gemini_pgvector config", exc_info=True)

    config["embedding_dimension"] = int(
        config.get("embedding_dimension", DEFAULT_EMBEDDING_DIMENSION)
    )
    config["top_k"] = max(1, int(config.get("top_k", DEFAULT_TOP_K)))
    config["request_timeout"] = max(1, int(config.get("request_timeout", 30)))
    config["max_memory_chars"] = max(
        1000, int(config.get("max_memory_chars", 12000))
    )
    config["auto_save"] = _as_bool(config.get("auto_save"), default=True)
    config["base_url"] = str(config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    return config


def _vector_to_sql(values: List[float]) -> str:
    return "[" + ",".join(format(value, ".12g") for value in values) + "]"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    pieces: List[str] = []
    for part in content:
        if isinstance(part, str):
            pieces.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
            elif part.get("type") in {"image", "image_url", "input_image"}:
                pieces.append("[image]")
    return "\n".join(pieces)


class GeminiPgvectorMemoryProvider(MemoryProvider):
    """Automatic semantic recall backed by Gemini Embedding 2 and pgvector."""

    def __init__(self) -> None:
        self._config: dict = {}
        self._api_key = ""
        self._database_url = ""
        self._base_url = DEFAULT_BASE_URL
        self._embedding_model = DEFAULT_EMBEDDING_MODEL
        self._embedding_dimension = DEFAULT_EMBEDDING_DIMENSION
        self._top_k = DEFAULT_TOP_K
        self._auto_save = True
        self._request_timeout = 30
        self._max_memory_chars = 12000
        self._user_id = "hermes-user"
        self._session_id = ""
        self._threads: List[threading.Thread] = []
        self._threads_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "gemini_pgvector"

    def is_available(self) -> bool:
        config = _load_config()
        if not config.get("api_key") or not config.get("database_url"):
            return False
        try:
            import psycopg  # noqa: F401
        except ImportError:
            return False
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "database_url",
                "description": "PostgreSQL connection URL",
                "secret": True,
                "required": True,
                "env_var": "DATABASE_URL",
            },
            {
                "key": "api_key",
                "description": "Google AI Studio API key",
                "secret": True,
                "required": True,
                "env_var": "GEMINI_API_KEY",
                "url": "https://aistudio.google.com/apikey",
            },
            {
                "key": "embedding_model",
                "description": "Gemini embedding model",
                "default": DEFAULT_EMBEDDING_MODEL,
            },
            {
                "key": "embedding_dimension",
                "description": "Embedding dimensions",
                "default": str(DEFAULT_EMBEDDING_DIMENSION),
            },
            {
                "key": "top_k",
                "description": "Memories recalled per message",
                "default": str(DEFAULT_TOP_K),
            },
            {
                "key": "auto_save",
                "description": "Automatically save completed turns",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        import yaml

        config_path = Path(hermes_home) / "config.yaml"
        root = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8-sig") as handle:
                root = yaml.safe_load(handle) or {}
        root.setdefault("memory", {})
        root["memory"]["gemini_pgvector"] = values
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(root, handle, sort_keys=False)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config["api_key"]
        self._database_url = self._config["database_url"]
        self._base_url = self._config["base_url"]
        self._embedding_model = self._config["embedding_model"]
        self._embedding_dimension = self._config["embedding_dimension"]
        self._top_k = self._config["top_k"]
        self._auto_save = self._config["auto_save"]
        self._request_timeout = self._config["request_timeout"]
        self._max_memory_chars = self._config["max_memory_chars"]
        self._session_id = session_id
        self._user_id = (
            str(kwargs.get("user_id") or "").strip()
            or str(self._config.get("user_id") or "").strip()
            or "hermes-user"
        )
        self._ensure_schema()

    def system_prompt_block(self) -> str:
        return (
            "# Gemini pgvector Memory\n"
            "Semantic memory is active. Relevant past context is recalled "
            "automatically before each response and completed turns are saved "
            "automatically."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query.strip():
            return ""
        try:
            memories = self._search_memories(query)
        except Exception as exc:
            logger.warning("Gemini pgvector recall failed: %s", exc)
            return ""
        if not memories:
            return ""

        lines = []
        for index, memory in enumerate(memories, 1):
            similarity = float(memory.get("similarity") or 0)
            lines.append(
                f"Memory {index} (similarity {similarity:.3f}): "
                f"{memory['content']}"
            )
        return "## Relevant Hermes Memory\n" + "\n\n".join(lines)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        user_text = _content_to_text(user_content).strip()
        assistant_text = _content_to_text(assistant_content).strip()
        if not self._auto_save or not user_text or not assistant_text:
            return
        content = (
            f"User said: {user_text}\n"
            f"Hermes replied: {assistant_text}"
        )[: self._max_memory_chars]
        self._start_background_save(content)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        if action == "add" and content.strip():
            self._start_background_save(content[: self._max_memory_chars])

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        with self._threads_lock:
            threads = list(self._threads)
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=5)

    def _connect(self):
        import psycopg

        return psycopg.connect(self._database_url, connect_timeout=10)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = %s::regclass
                      AND attname = 'embedding'
                      AND NOT attisdropped
                    """,
                    (TABLE_NAME,),
                )
                row = cursor.fetchone()
                if not row:
                    raise RuntimeError(f"Required table {TABLE_NAME} is missing")
                if int(row[0]) != self._embedding_dimension:
                    raise RuntimeError(
                        f"{TABLE_NAME}.embedding dimension mismatch: "
                        f"expected {self._embedding_dimension}, got {row[0]}"
                    )

    def _create_embedding(self, text: str, embedding_type: str) -> List[float]:
        prefix = (
            "task: retrieval_query | query: "
            if embedding_type == "query"
            else "task: retrieval_document | content: "
        )
        model = quote(self._embedding_model, safe="")
        url = f"{self._base_url}/models/{model}:embedContent?key={quote(self._api_key)}"
        payload = {
            "content": {"parts": [{"text": prefix + text}]},
            "outputDimensionality": self._embedding_dimension,
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._request_timeout) as response:
                body = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini embedding HTTP {exc.code}: {details[:500]}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

        embedding = body.get("embedding")
        if not embedding and body.get("embeddings"):
            embedding = body["embeddings"][0]
        values = (embedding or {}).get("values")
        if not values:
            raise RuntimeError("Gemini embedding response did not contain values")
        if len(values) != self._embedding_dimension:
            raise RuntimeError(
                "Gemini embedding dimension mismatch: "
                f"expected {self._embedding_dimension}, got {len(values)}"
            )
        return [float(value) for value in values]

    def _save_memory(self, content: str) -> None:
        embedding = self._create_embedding(content, "document")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (user_id, content, embedding)
                    VALUES (%s, %s, %s::extensions.vector)
                    """,
                    (self._user_id, content, _vector_to_sql(embedding)),
                )

    def _search_memories(self, query: str) -> List[dict]:
        embedding = self._create_embedding(query, "query")
        vector = _vector_to_sql(embedding)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH query_embedding AS (
                        SELECT %s::extensions.vector AS value
                    )
                    SELECT content,
                           1 - (embedding <=> query_embedding.value) AS similarity
                    FROM {TABLE_NAME}, query_embedding
                    WHERE user_id = %s
                    ORDER BY embedding <=> query_embedding.value
                    LIMIT %s
                    """,
                    (vector, self._user_id, self._top_k),
                )
                return [
                    {"content": row[0], "similarity": float(row[1])}
                    for row in cursor.fetchall()
                ]

    def _start_background_save(self, content: str) -> None:
        def _run() -> None:
            try:
                self._save_memory(content)
            except Exception as exc:
                logger.warning("Gemini pgvector memory save failed: %s", exc)
            finally:
                current = threading.current_thread()
                with self._threads_lock:
                    self._threads = [
                        thread for thread in self._threads if thread is not current
                    ]

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name="gemini-pgvector-save",
        )
        with self._threads_lock:
            self._threads.append(thread)
        thread.start()


def register(ctx) -> None:
    ctx.register_memory_provider(GeminiPgvectorMemoryProvider())
