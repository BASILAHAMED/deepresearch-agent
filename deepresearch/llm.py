"""LLM utility — thin wrapper supporting OpenAI and Anthropic."""

from __future__ import annotations

import json
import re
from deepresearch.config import Config


def get_llm_client(config: Config):
    """Return an (invoke_fn, provider_name) tuple."""
    if config.llm_provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        def invoke(system: str, user: str, temperature: float = 0.3) -> str:
            resp = client.messages.create(
                model=config.llm_model,
                max_tokens=4096,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        return invoke, "anthropic"
    else:
        from openai import OpenAI
        client = OpenAI(api_key=config.openai_api_key)

        def invoke(system: str, user: str, temperature: float = 0.3) -> str:
            resp = client.chat.completions.create(
                model=config.llm_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

        return invoke, "openai"


def parse_json_response(text: str) -> dict | list:
    """Extract JSON from an LLM response that may contain markdown fences."""
    # Try direct parse first
    text = text.strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Extract from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Last resort: find first { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")
