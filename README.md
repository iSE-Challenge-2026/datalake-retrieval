# Data-Lake Retrieval

This repository builds the retrieval layer for a multimodal Data-Lake QA system.
It normalizes raw files into canonical text/image/table artifacts, builds chunks
and embeddings, then evaluates retrieval of the source files needed by a
question-answering repo.

The current default retrieval config is vector-only plus Gemini 2.5 Flash
reranking:

```text
preset = vector_flash
retrieval_mode = vector
vector_chunk_k = 80
vector_aggregation = legacy
rerank_mode = llm
rerank_model = google/gemini-2.5-flash
expand_mentioned_folders = false
no_special_cases = true
```

The alternative high-recall preset is:

```text
folder_expand_flash
```

It uses the same vector retrieval and reranker, then expands explicitly mentioned
folders when needed.

## Repository Layout

```text
configs/pipeline.yaml              Single project config
src/                               Pipeline implementation
scripts/                           Thin CLI wrappers
data/raw/Data-Lake/                Raw challenge data lake
data/processed/Data-Lake/          Reusable processed artifacts
data/output/                       Retrieval runs, reports, handoff bundles
```

Processed artifacts are stored here:

```text
data/processed/Data-Lake/canonical/
data/processed/Data-Lake/model_raw/datalab_parsing/
data/processed/Data-Lake/model_raw/openrouter_asr/
data/processed/Data-Lake/model_raw/openrouter_embeddings/
data/processed/Data-Lake/model_raw/openrouter_image_enrichment/
data/processed/Data-Lake/model_raw/openrouter_table_enrichment/
```

Retrieval/evaluation outputs are stored here:

```text
data/output/benchmarks/
data/output/reports/
data/output/rerank_cache/
data/output/handoff/
```

## Install

Python 3.11+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the package and dependencies with pip:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

If you want extra dataframe/parquet tooling for your own experiments, install
these too:

```bash
python -m pip install pandas openpyxl pyarrow
```

## Environment

Copy the environment template:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Fill the keys you need:

```dotenv
DATALAB_API_KEY=
OPENROUTER_API_KEY=

OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_RERANK_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_REASONING_MODEL=google/gemini-2.5-flash
OPENROUTER_VISION_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_TABLE_DESCRIPTION_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_ASR_MODEL=openai/whisper-large-v3
```

`DATALAB_API_KEY` is used for Lift/Datalab parsing. `OPENROUTER_API_KEY` is used
for ASR, enrichment, embeddings, and LLM reranking.
Retrieval benchmark presets are controlled by `configs/pipeline.yaml`; `.env`
model names are fallbacks and role-specific defaults.

## Run The Pipeline

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

This runs:

```text
canonical -> image_enrichment -> table_enrichment -> embeddings -> audit
```

Run only embeddings after canonical artifacts change:

```bash
python scripts/run_pipeline.py --stages embeddings
```

Run individual stages when debugging:

```bash
python scripts/build_canonical_artifacts.py
python scripts/enrich_image_descriptions.py
python scripts/enrich_table_descriptions.py
python scripts/build_canonical_embeddings.py
python scripts/audit_canonical_quality.py
```

## Retrieval Evaluation

Run the default vector-only preset:

Windows PowerShell:

```powershell
python scripts/retrieve_eval.py ^
  --questions data/0.Sample_Data.xlsx ^
  --preset vector_flash ^
  --bench-dir data/output/benchmarks/rerank_variants ^
  --run-name vector_flash
```

macOS/Linux:

```bash
python scripts/retrieve_eval.py \
  --questions data/0.Sample_Data.xlsx \
  --preset vector_flash \
  --bench-dir data/output/benchmarks/rerank_variants \
  --run-name vector_flash
```

Run the folder-expansion variant:

Windows PowerShell:

```powershell
python scripts/retrieve_eval.py ^
  --questions data/0.Sample_Data.xlsx ^
  --preset folder_expand_flash ^
  --bench-dir data/output/benchmarks/rerank_variants ^
  --run-name folder_expand_flash
```

macOS/Linux:

```bash
python scripts/retrieve_eval.py \
  --questions data/0.Sample_Data.xlsx \
  --preset folder_expand_flash \
  --bench-dir data/output/benchmarks/rerank_variants \
  --run-name folder_expand_flash
```

Outputs include:

```text
manifest.json
summary.json
results.json
```

## Handoff To QA Repo

For another QA repo, the most useful files are:

```text
data/processed/Data-Lake/canonical/texts.jsonl
data/processed/Data-Lake/canonical/images.jsonl
data/processed/Data-Lake/canonical/tables.jsonl
data/processed/Data-Lake/canonical/chunks.jsonl
data/processed/Data-Lake/model_raw/openrouter_embeddings/vector_records.jsonl
```

Export a compact handoff bundle:

```bash
python scripts/export_handoff_bundle.py --overwrite
```

Embeddings are stored under `model_raw/openrouter_embeddings` instead of
`canonical` because they are model-specific cache artifacts. Canonical files
should stay model-agnostic and stable.

## Tests

```bash
python -m unittest discover tests
```
