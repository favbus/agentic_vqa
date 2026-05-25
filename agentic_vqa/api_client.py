from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def merged_env(path: Path = Path(".env")) -> dict[str, str]:
    env = dict(os.environ)
    env.update(load_dotenv(path))
    return env


def required_env(env: dict[str, str], name: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing. Set it in .env or the process environment.")
    return value


class RelayClient:
    """Small OpenAI-compatible relay client matching the project's existing intermediary style."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        api_style: str = "auto",
        request_timeout: float = 240.0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.request_timeout = request_timeout
        self.reasoning_effort = reasoning_effort
        api_style = api_style.strip().lower()
        if api_style not in {"auto", "responses", "chat"}:
            raise ValueError("api_style must be one of: auto, responses, chat")
        self.api_style = api_style
        self._prefer_chat = api_style == "chat" or (api_style == "auto" and "api.asxs.top" in self.base_url)

    def create(
        self,
        input_items: list[dict[str, Any]],
        *,
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
        retries: int = 3,
        retry_sleep: float = 5.0,
    ) -> dict[str, Any]:
        if self._prefer_chat:
            return self._create_chat(input_items, max_output_tokens, temperature, retries, retry_sleep)
        return self._create_responses(input_items, max_output_tokens, temperature, retries, retry_sleep)

    def _create_responses(
        self,
        input_items: list[dict[str, Any]],
        max_output_tokens: int,
        temperature: float,
        retries: int,
        retry_sleep: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                self.base_url + "/responses",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "OpenAI/Python 1.0.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if self.api_style == "auto" and data.get("status") == "completed" and not data.get("output"):
                        self._prefer_chat = True
                        return self._create_chat(input_items, max_output_tokens, temperature, retries, retry_sleep)
                    return data
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:1000]}")
                if exc.code in {400, 401, 403, 404}:
                    raise last_error from exc
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent.
                last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
        raise RuntimeError(f"Responses API failed after {retries} attempts: {last_error}")

    def _create_chat(
        self,
        input_items: list[dict[str, Any]],
        max_output_tokens: int,
        temperature: float,
        retries: int,
        retry_sleep: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._responses_input_to_chat_messages(input_items),
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "OpenAI/Python 1.0.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    return json.loads(resp.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:1000]}")
                if exc.code in {400, 401, 403, 404}:
                    raise last_error from exc
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent.
                last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
        raise RuntimeError(f"Chat Completions API failed after {retries} attempts: {last_error}")

    @staticmethod
    def _responses_input_to_chat_messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in input_items:
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            chat_content = []
            for part in content:
                part_type = part.get("type")
                if part_type in {"input_text", "output_text", "text"}:
                    chat_content.append({"type": "text", "text": str(part.get("text", ""))})
                elif part_type in {"input_image", "image_url"}:
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    chat_content.append({"type": "image_url", "image_url": {"url": str(image_url)}})
            messages.append({"role": role, "content": chat_content})
        return messages


def response_text(payload: dict[str, Any]) -> str:
    if "choices" in payload:
        chunks: list[str] = []
        for choice in payload.get("choices") or []:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") in {"text", "output_text"}:
                        chunks.append(str(part.get("text", "")))
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    chunks = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty model output")
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found")
    return json.loads(text[first : last + 1])

