#!/usr/bin/env python3
"""Run a batch of prompts against OpenRouter via the OpenAI SDK."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm
import yaml


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    base_url: str
    headers: Dict[str, str]
    parameters: Dict[str, Any]
    system_prompt: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompts against OpenRouter.")
    parser.add_argument("--prompts", required=True, type=Path, help="Path to prompts JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output JSONL file.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configuration JSON file.")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent execution threads (default: 4).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                prompts.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def _load_raw_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data: Optional[Dict[str, Any]] = None

    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must define a mapping at the top level.")
    return data


def load_config(path: Path) -> RunConfig:
    raw_config = _load_raw_config(path)

    name = raw_config.get("name")
    model = raw_config.get("model")
    if not name or not model:
        raise ValueError("Config must include non-empty 'name' and 'model' fields.")

    base_url = raw_config.get("base_url", DEFAULT_BASE_URL)
    headers = raw_config.get("headers", {}) or {}
    parameters = raw_config.get("parameters", {}) or {}
    system_prompt = raw_config.get("system_prompt")

    default_headers = {
        "HTTP-Referer": headers.get("HTTP-Referer") or "https://github.com/andrewplassard/llm-semantic-analysis",
        "X-Title": headers.get("X-Title") or name,
    }
    merged_headers = {**headers, **default_headers}

    return RunConfig(
        name=name,
        model=model,
        base_url=base_url,
        headers=merged_headers,
        parameters=parameters,
        system_prompt=system_prompt,
    )


def create_client(config: RunConfig) -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key, base_url=config.base_url, default_headers=config.headers)


def build_input(prompt_text: str, system_prompt: Optional[str]) -> List[Dict[str, Any]]:
    dialogue: List[Dict[str, Any]] = []
    if system_prompt:
        dialogue.append({"role": "system", "content": system_prompt})
    dialogue.append({"role": "user", "content": prompt_text})
    return dialogue


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    chunks: List[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            content_text = getattr(content, "text", None)
            if content_text:
                chunks.append(content_text)
    return "\n".join(chunks).strip()


def run_single_prompt(
    prompt: Dict[str, Any],
    client: OpenAI,
    config: RunConfig,
) -> Dict[str, Any]:
    request_parameters = dict(config.parameters)
    system_prompt = request_parameters.pop("system_prompt", None) or config.system_prompt

    input_payload = build_input(prompt.get("text", ""), system_prompt)

    response = client.responses.create(
        model=config.model,
        input=input_payload,
        **request_parameters,
    )

    output_text = extract_output_text(response)

    result = {
        "run_name": config.name,
        "model": config.model,
        "prompt": prompt,
        "response_text": output_text,
    }

    usage = getattr(response, "usage", None)
    if usage:
        result["usage"] = usage.model_dump()

    result["response"] = response.model_dump()
    return result


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    prompts_path: Path = args.prompts
    output_path: Path = args.output
    config_path: Path = args.config

    prompts = load_jsonl(prompts_path)
    config = load_config(config_path)
    client = create_client(config)

    threads = max(1, args.threads)
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(run_single_prompt, prompt, client, config): idx
            for idx, prompt in enumerate(prompts)
        }

        with tqdm(total=len(prompts), desc=f"Run {config.name}", unit="prompt") as progress:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[idx] = {
                        "run_name": config.name,
                        "model": config.model,
                        "prompt": prompts[idx],
                        "error": str(exc),
                    }
                finally:
                    progress.update(1)

    assert all(result is not None for result in results), "Missing results for one or more prompts."
    write_jsonl(output_path, [result for result in results if result is not None])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
