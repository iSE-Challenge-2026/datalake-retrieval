# Data-Lake Retrieval

This repository contains the retrieval pipeline for the multimodal Data-Lake QA
challenge. It turns the raw Data-Lake into canonical artifacts, builds chunks and
embeddings, and benchmarks source-file retrieval for downstream question
answering.

The default retrieval setup is:

```text
preset: vector_flash
retrieval: vector-only
reranker: google/gemini-2.5-flash
folder expansion: off
special rules: off
```

Use `folder_expand_flash` only when you explicitly want the high-recall variant
that expands mentioned folders.

## Quick Start

1. Create a Python 3.11+ environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

3. Create `.env`.

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Fill at least:

```dotenv
DATALAB_API_KEY=
OPENROUTER_API_KEY=
```

4. Put the raw Data-Lake here:

```text
data/raw/Data-Lake/
```

5. Run the full processing pipeline.

```bash
python scripts/run_pipeline.py
```

6. Run the default retrieval benchmark.

```bash
python scripts/retrieve_eval.py --questions data/0.Sample_Data.xlsx --preset vector_flash --run-name vector_flash
```

Benchmark output is written to:

```text
data/output/Data-Lake/benchmarks/
```

7. Export retrieved source files for downstream QA/reasoning.

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset vector_flash
```

The default output is:

```text
data/output/Data-Lake/retrieval_source_paths.json
```

## What The Pipeline Produces

Reusable processed artifacts are written under `data/processed/Data-Lake`:

```text
data/processed/Data-Lake/canonical/
  texts.jsonl
  images.jsonl
  tables.jsonl
  chunks.jsonl

data/processed/Data-Lake/model_raw/
  datalab_parsing/
  openrouter_asr/
  openrouter_embeddings/
  openrouter_image_enrichment/
  openrouter_table_enrichment/
```

Final retrieval outputs are written under `data/output/Data-Lake`:

```text
data/output/Data-Lake/benchmarks/
data/output/Data-Lake/reports/
data/output/Data-Lake/rerank_cache/
data/output/Data-Lake/handoff/
```

The main files to hand off to a QA/reasoning repo are:

```text
data/processed/Data-Lake/canonical/texts.jsonl
data/processed/Data-Lake/canonical/images.jsonl
data/processed/Data-Lake/canonical/tables.jsonl
data/processed/Data-Lake/canonical/chunks.jsonl
data/processed/Data-Lake/model_raw/openrouter_embeddings/vector_records.jsonl
```

Embeddings are stored in `model_raw/openrouter_embeddings` instead of
`canonical` because embeddings are model-specific cache artifacts. Canonical data
should stay model-agnostic.

## Common Commands

Run everything:

```bash
python scripts/run_pipeline.py
```

The default full pipeline order is:

```text
canonical -> image_enrichment -> table_enrichment -> embeddings -> audit
```

Do not skip `table_enrichment` before rebuilding embeddings if you want table
retrieval quality to match the benchmarked setup. The SQL/table descriptions are
merged into `tables.jsonl` and then embedded.

Run only specific stages:

```bash
python scripts/run_pipeline.py --stages canonical
python scripts/run_pipeline.py --stages image_enrichment,table_enrichment
python scripts/run_pipeline.py --stages embeddings
python scripts/run_pipeline.py --stages audit
```

Run individual wrappers when debugging:

```bash
python scripts/build_canonical_artifacts.py
python scripts/enrich_image_descriptions.py
python scripts/enrich_table_descriptions.py
python scripts/build_canonical_embeddings.py
python scripts/audit_canonical_quality.py
```

Run default vector-only retrieval:

```bash
python scripts/retrieve_eval.py --questions data/0.Sample_Data.xlsx --preset vector_flash --run-name vector_flash
```

Run high-recall folder expansion:

```bash
python scripts/retrieve_eval.py --questions data/0.Sample_Data.xlsx --preset folder_expand_flash --run-name folder_expand_flash
```

Export source paths for every question:

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset vector_flash
```

Export the folder-expansion variant:

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset folder_expand_flash --output data/output/Data-Lake/retrieval_source_paths_folder_expand.json
```

If you need a CSV instead of JSON:

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset vector_flash --output-format csv --output data/output/Data-Lake/retrieval_source_paths.csv
```

If processed artifacts do not exist yet, run ingestion/indexing first:

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset vector_flash --build-first
```

Export a compact bundle for another repo:

```bash
python scripts/export_handoff_bundle.py --overwrite
```

Run tests:

```bash
python -m unittest discover tests
```

## Configuration

The single project config is:

```text
configs/pipeline.yaml
```

Important retrieval defaults:

```text
default_preset = vector_flash
retrieval_mode = vector
vector_chunk_k = 80
vector_aggregation = legacy
rerank_mode = llm
rerank_model = google/gemini-2.5-flash
expand_mentioned_folders = false
no_special_cases = true
```

Important parsing defaults:

```text
global Datalab/Lift operation = convert
global Datalab/Lift mode = fast
raw image Lift operation = extract
raw image Lift mode = fast
```

Document files such as PDF/PPT/PPTX use existing Datalab parsing artifacts when
available. Raw images are parsed through Lift extract, then image descriptions
can be enriched by OpenRouter.

`.env` contains API keys and model fallbacks. Presets in
`configs/pipeline.yaml` take priority for benchmark runs.

## Notes For Collaborators

- `data/raw/`, `data/processed/`, and `data/output/` are ignored by git.
- If someone gives you a processed artifact bundle, place it under
  `data/processed/Data-Lake/` and you can run retrieval without rebuilding the
  full ingestion pipeline.
- `scripts/run_solution.py` is the handoff entrypoint for other teams. It reads a
  question file, runs the configured retrieval preset, and writes local absolute
  source file paths for each question.
- If `chunks.jsonl` changes, rerun embeddings:

```bash
python scripts/run_pipeline.py --stages embeddings
```

- LLM rerank cache lives in:

```text
data/output/Data-Lake/rerank_cache/
```

- Benchmark runs contain:

```text
manifest.json
summary.json
results.json
```

