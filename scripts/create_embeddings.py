#!/usr/bin/env python3
"""Create embedding parquet files from LLM run results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings from an LLM results JSONL file.")
    parser.add_argument("--input", type=Path, help="Path to a results JSONL file.")
    parser.add_argument("--inputs", type=Path, nargs="+", help="One or more results JSONL files.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to a specific embeddings parquet. Defaults to embeddings/<input_basename>.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding inference.")
    parser.add_argument("--device", default="cpu", help="Device for the embedding model (e.g. cpu, cuda).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def extract_response_text(record: Dict[str, Any]) -> Optional[str]:
    text = record.get("response_text")
    if text:
        return text

    response = record.get("response")
    if not response:
        return None

    text = response.get("output_text")
    if text:
        return text

    for output in response.get("output", []) or []:
        if output.get("type") == "message":
            for content in output.get("content", []) or []:
                content_text = content.get("text")
                if content_text:
                    return content_text
        content = output.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def default_output_path(input_path: Path) -> Path:
    output_dir = Path("embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}.parquet"


def process_file(
    input_path: Path,
    output_path: Path,
    model: SentenceTransformer,
    batch_size: int,
) -> None:
    records = load_jsonl(input_path)
    rows: List[Dict[str, Any]] = []
    texts: List[str] = []

    for record in records:
        if record.get("error"):
            continue
        response_text = extract_response_text(record)
        if not response_text:
            continue
        prompt_data = record.get("prompt") or {}
        rows.append(
            {
                "source_file": str(input_path),
                "run_name": record.get("run_name"),
                "model": record.get("model"),
                "prompt_id": prompt_data.get("id"),
                "prompt_text": prompt_data.get("text"),
                "response_text": response_text,
            }
        )
        texts.append(response_text)

    if not rows:
        raise ValueError(f"No rows with usable responses found in {input_path}")

    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    for row, vector in zip(rows, embeddings):
        row["embedding"] = vector.tolist()

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} embeddings to {output_path}")


def main() -> None:
    args = parse_args()

    if args.inputs:
        input_paths = list(args.inputs)
    elif args.input:
        input_paths = [args.input]
    else:
        raise SystemExit("Please provide --input or --inputs.")

    if len(input_paths) > 1 and args.output:
        raise SystemExit("Cannot use --output with multiple inputs.")

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=args.device)

    for path in input_paths:
        output_path = args.output if args.output else default_output_path(path)
        if output_path.exists() and not args.force:
            print(f"Skipping {path}: {output_path} already exists (use --force to overwrite).")
            continue
        try:
            process_file(path, output_path, model, args.batch_size)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")


if __name__ == "__main__":
    main()
