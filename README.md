# Data-Lake Retrieval

Retrieval pipeline for the Data-Lake QA challenge. This repo builds processed
artifacts and exports source-file paths for the downstream `da-agent` repo.

Expected layout:

```text
iSE Summer Challenge 2026/
  datalake-retrieval/
  da-agent/
```

## Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Create `.env` from `.env.example` and fill:

```dotenv
DATALAB_API_KEY=
OPENROUTER_API_KEY=
```

Put raw data here:

```text
data/raw/Data-Lake/
```

## Run

If artifacts are already built, export source paths:

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset vector_flash
```

If this is the first run, build artifacts first:

```bash
python scripts/run_solution.py --questions data/0.Sample_Data.xlsx --preset vector_flash --build-first
```

Output:

```text
data/output/Data-Lake/retrieval_source_paths.json
```

This file is the retrieval input for `da-agent`.

## Important Files

```text
configs/pipeline.yaml
data/processed/Data-Lake/canonical/chunks.jsonl
data/processed/Data-Lake/model_raw/openrouter_embeddings/vector_records.jsonl
data/output/Data-Lake/retrieval_source_paths.json
```

`data/raw/`, `data/processed/`, and `data/output/` are ignored by git.

