"""Build canonical chunks and embeddings for retrieval benchmarking."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical.embedding import (  # noqa: E402
    ChunkBuildConfig,
    build_canonical_chunks,
    build_vector_records,
    write_chunks,
    write_vector_outputs,
)
from src.model_clients.embeddings import OpenRouterEmbeddingProvider  # noqa: E402
from src.project_paths import CANONICAL_CHUNKS_PATH, CANONICAL_DIR, OPENROUTER_EMBEDDINGS_DIR  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv_file(PROJECT_ROOT)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    known, _ = bootstrap.parse_known_args(argv)
    pipeline_config = _load_pipeline_config(_resolve(known.config))
    defaults = pipeline_config.get("canonical_embeddings", {})

    parser = argparse.ArgumentParser(description="Chunk canonical text/image artifacts and build embeddings.", parents=[bootstrap])
    parser.add_argument("--canonical-dir", type=Path, default=Path(defaults.get("canonical_dir", CANONICAL_DIR)))
    parser.add_argument("--chunks-path", type=Path, default=Path(defaults.get("chunks_path", CANONICAL_CHUNKS_PATH)))
    parser.add_argument("--embedding-output-dir", type=Path, default=Path(defaults.get("embedding_output_dir", OPENROUTER_EMBEDDINGS_DIR)))
    parser.add_argument("--model", default=defaults.get("model") or os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"))
    parser.add_argument("--dimension", type=int, default=int(defaults.get("dimension") or os.getenv("OPENROUTER_EMBEDDING_DIMENSION", "1536")))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 32)))
    parser.add_argument("--chunk-size", type=int, default=int(defaults.get("chunk_size", 1200)))
    parser.add_argument("--chunk-overlap", type=int, default=int(defaults.get("chunk_overlap", 150)))
    parser.add_argument("--chunks-only", action="store_true")
    args = parser.parse_args(argv)

    chunks, chunk_report = build_canonical_chunks(
        ChunkBuildConfig(
            canonical_dir=_resolve(args.canonical_dir),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )
    chunks_path = _resolve(args.chunks_path)
    write_chunks(chunks, chunks_path, report=chunk_report)

    payload = {
        "chunks": chunk_report,
        "chunks_path": chunks_path.as_posix(),
    }
    if not args.chunks_only:
        provider = OpenRouterEmbeddingProvider(
            model=args.model,
            dimension=args.dimension,
            api_key_env="OPENROUTER_API_KEY",
            app_title="Data-Lake Canonical Retrieval",
        )
        output_dir = _resolve(args.embedding_output_dir)
        vectors, embedding_report = build_vector_records(chunks, provider, output_dir, batch_size=args.batch_size)
        write_vector_outputs(vectors, embedding_report, output_dir)
        payload["embeddings"] = embedding_report
        payload["vector_records_path"] = (output_dir / "vector_records.jsonl").as_posix()

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_pipeline_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
