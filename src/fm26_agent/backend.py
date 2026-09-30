from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from .config import LLMSettings


@dataclass
class ChatReply:
    message: dict[str, Any]
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


class ChatBackend(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatReply: ...


class OpenAICompatibleBackend:
    def __init__(self, settings: LLMSettings, api_key: str, client: Any = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=settings.base_url, timeout=90, max_retries=2)
        self.client = client
        self.settings = settings

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatReply:
        options: dict[str, Any] = {"max_tokens": self.settings.max_output_tokens}
        if not tools:
            options["response_format"] = {"type": "json_object"}
        if urlparse(self.settings.base_url).hostname == "api.deepseek.com":
            options["extra_body"] = {
                "thinking": {"type": "enabled" if self.settings.thinking else "disabled"}
            }
        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            tools=tools or None,
            temperature=self.settings.temperature,
            **options,
        )
        # Preserve reasoning_content and tool-call IDs if the provider returns them.
        message = response.choices[0].message.model_dump(exclude_none=True)
        message.pop("refusal", None)
        message.pop("annotations", None)
        usage = response.usage.model_dump() if response.usage else {}
        return ChatReply(
            message,
            {key: value for key, value in usage.items() if isinstance(value, int)},
            getattr(response.choices[0], "finish_reason", None),
        )
