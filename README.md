# AI ETL Pipeline

An AI-powered ETL (Extract, Transform, Load) pipeline that uses a local LLM via Ollama to automatically clean any CSV or JSON dataset. The agent analyzes the data profile and decides which cleaning operations to apply.

## How it works

1. **Extract** — reads a CSV or JSON file into a pandas DataFrame
2. **Transform** — runs deterministic pre-cleaning, then hands the data profile to an AI agent (Ollama) which calls cleaning tools to fix issues it finds. If a DuckDB destination table is provided, its schema is passed to the agent as a style guide and enforced deterministically afterwards
3. **Load** — saves the cleaned DataFrame to a new CSV file, and optionally appends/creates/replaces a table in a DuckDB database

The AI agent does not receive the full dataset — it sees a profile (column names, data types, null percentages) that includes **up to 5 sample values per column**. Those samples are real rows, so the model does see a small slice of raw data (names, emails, etc.). With the default local Ollama model nothing leaves your machine, but keep this in mind before pointing the pipeline at a cloud API. The agent decides which tools to call based on what it finds in the profile.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- The `qwen2.5:7b` model pulled in Ollama

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install and start Ollama
# Download from https://ollama.com, then:
ollama pull qwen2.5:7b
ollama serve
```

## Running

### Option 1 — Streamlit frontend (recommended)

```bash
streamlit run app.py
```

Opens a browser UI where you can upload any CSV, run the pipeline, preview the cleaned data, and download the result. You can also optionally load the cleaned data into a DuckDB database in the `data/` folder (appending to an existing table requires matching columns), and browse/export existing DuckDB tables at the bottom of the page.

### Option 2 — Command line

```bash
python etl_pipeline.py your_file.csv
# or
python etl_pipeline.py your_file.json
```

Saves output to `cleaned_output.csv` in the same directory.

## Project structure

```
etl_pipeline/
├── etl_pipeline.py   # Core pipeline logic and AI agent
├── app.py            # Streamlit frontend
├── tests/            # Unit tests (pytest)
├── data/             # DuckDB database files created by the app
├── audit/            # Per-run audit logs (gitignored)
├── requirements.txt  # Python dependencies
└── README.md
```

## Running the tests

```bash
python -m pytest tests/
```

## Cleaning operations the agent can perform

| Tool | What it does |
|---|---|
| `drop_column` | Drops a column from the dataset |
| `convert_dtype` | Converts a column to float, int, str, or Int64 (nullable integer) |
| `rename_column` | Renames a column (e.g. removes spaces or special characters) |
| `strip_characters` | Strips leading/trailing characters (e.g. `$`, `,`) |
| `standardize_case` | Converts all string values to lower or upper case |
| `drop_duplicates` | Removes identical duplicate rows |
| `standardize_date` | Normalizes dates to YYYY-MM-DD from MM/DD/YYYY, MM-DD-YYYY, etc. |
| `standardize_phone` | Formats phone numbers to XXX-XXX-XXXX |
| `validate_email` | Replaces invalid email addresses with empty string |
| `replace_nan_strings` | Replaces placeholder strings ("nan", "N/A", "none", "null", "unknown", any casing) with empty string |
| `flag_negative_balance` | Adds a `<column>_negative_flag` column marking rows where a numeric column is negative |
| `trim_whitespace` | Strips leading/trailing spaces from string values |

## Deterministic pre-cleaning (always runs before the agent)

These steps run in Python before the agent is invoked — guaranteed to work regardless of model quality:

- Duplicate rows removed
- Placeholder strings ("nan", "N/A", "none", "null", "unknown") replaced with empty string

Case normalization is left to the agent, which standardizes categorical columns per column rather than blanket-lowercasing everything (which would destroy name/address casing).

## Changing the model

The model is set at the top of `etl_pipeline.py`:

```python
MODEL = "qwen2.5:7b"
```

Larger models follow instructions more reliably:

| Model | Size | Notes |
|---|---|---|
| `qwen2.5:3b` | 3B | Faster, lower accuracy — for low-RAM machines |
| `qwen2.5:7b` | 7B | Default — better instruction following, needs ~8GB RAM |
| `llama3.1:8b` | 8B | Good alternative if qwen2.5 is unavailable |

To switch:
```bash
ollama pull qwen2.5:3b
# then update MODEL = "qwen2.5:3b" in etl_pipeline.py
```

---

## For Claude

- **Entry point:** `etl_pipeline.py` — contains `extract`, `transform`, `load`, all cleaning functions, tool definitions, and the agent loop
- **Frontend:** `app.py` — Streamlit UI that calls `extract`, `transform`, `load` from `etl_pipeline.py`
- **Agent loop:** `run_agent()` in `etl_pipeline.py` — uses the OpenAI-compatible Ollama API at `http://localhost:11434/v1`
- **Tool definitions:** `tools` list in `etl_pipeline.py` in OpenAI function-calling format (`{"type": "function", "function": {...}}`)
- **Tool dispatcher:** `run_tool()` maps tool call names from the agent to the corresponding Python functions
- **Data profile:** `get_data_profile()` builds the column summary passed to the agent — adding new fields here gives the agent more signal to make better decisions
- **Pre-cleaning:** `transform()` runs `drop_duplicates` and `replace_nan_strings` on a copy of the input before calling `run_agent()` — these are deterministic and not delegated to the model
