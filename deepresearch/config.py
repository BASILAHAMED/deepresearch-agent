"""Configuration and environment setup."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Runtime configuration loaded from environment."""

    serpapi_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = ""
    max_iterations: int = 3  # max plan→search→evaluate loops
    max_sources_per_query: int = 10
    search_engines: list = field(default_factory=lambda: [
        "google", "google_scholar", "google_news"
    ])

    def __post_init__(self):
        self.serpapi_key = self.serpapi_key or os.getenv("SERPAPI_KEY", "")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)

        if not self.llm_model:
            if self.llm_provider == "anthropic":
                self.llm_model = "claude-haiku-5-5"
            else:
                self.llm_model = "gpt-4o-mini"

    def validate(self) -> list[str]:
        errors = []
        if not self.serpapi_key:
            errors.append("SERPAPI_KEY is required")
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when using OpenAI")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic")
        return errors
