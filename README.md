# AI ETL Pipeline

An AI-powered ETL (Extract, Transform, Load) pipeline that uses a local LLM via Ollama to automatically clean any CSV or JSON dataset. The agent analyzes the data profile and decides which cleaning operations to apply.

## How it works

1. **Extract** — reads a CSV or JSON file into a pandas DataFrame
2. **Transform** — runs deterministic pre-cleaning, then hands the data profile to an AI agent (Ollama) which calls cleaning tools to fix issues it finds
3. **Load** — saves the cleaned DataFrame to a new CSV file

The AI agent never sees the raw data — it only sees a profile (column names, data types, null percentages, sample values). It then decides which tools to call based on what it finds.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- The `qwen2.5:3b` model pulled in Ollama

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install and start Ollama
# Download from https://ollama.com, then:
ollama pull qwen2.5:3b
ollama serve
```

## Running

### Option 1 — Streamlit frontend (recommended)

```bash
streamlit run app.py
```

Opens a browser UI where you can upload any CSV, run the pipeline, preview the cleaned data, and download the result.

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
├── requirements.txt  # Python dependencies
└── README.md
```

## Cleaning operations the agent can perform

| Tool | What it does |
|---|---|
| `drop_column` | Drops a column from the dataset |
| `fill_nulls` | Fills missing values using mean, median, mode, or a fixed value |
| `convert_dtype` | Converts a column to float, int, or str |
| `rename_column` | Renames a column (e.g. removes spaces or special characters) |
| `strip_characters` | Strips leading/trailing characters (e.g. `$`, `,`) |
| `standardize_case` | Converts all string values to lower or upper case |
| `drop_duplicates` | Removes identical duplicate rows |
| `standardize_date` | Normalizes dates to YYYY-MM-DD from MM/DD/YYYY, MM-DD-YYYY, etc. |
| `standardize_phone` | Formats phone numbers to XXX-XXX-XXXX |
| `validate_email` | Replaces invalid email addresses with empty string |
| `replace_nan_strings` | Replaces literal "nan" strings with empty string |
| `flag_negative_balance` | Adds a `balance_flag` column marking negative balance rows |

## Deterministic pre-cleaning (always runs before the agent)

These steps run in Python before the agent is invoked — guaranteed to work regardless of model quality:

- All string columns lowercased
- Duplicate rows removed
- Literal "nan" strings replaced with empty string

## Changing the model

The model is set at the top of `etl_pipeline.py`:

```python
MODEL = "qwen2.5:3b"
```

Larger models follow instructions more reliably:

| Model | Size | Notes |
|---|---|---|
| `qwen2.5:3b` | 3B | Default — fast, lower accuracy |
| `qwen2.5:7b` | 7B | Better instruction following, needs ~8GB RAM |
| `llama3.1:8b` | 8B | Good alternative if qwen2.5 is unavailable |

To switch:
```bash
ollama pull qwen2.5:7b
# then update MODEL = "qwen2.5:7b" in etl_pipeline.py
```

---

## For Claude

- **Entry point:** `etl_pipeline.py` — contains `extract`, `transform`, `load`, all cleaning functions, tool definitions, and the agent loop
- **Frontend:** `app.py` — Streamlit UI that calls `extract`, `transform`, `load` from `etl_pipeline.py`
- **Agent loop:** `run_agent()` in `etl_pipeline.py` — uses the OpenAI-compatible Ollama API at `http://localhost:11434/v1`
- **Tool definitions:** `tools` list in `etl_pipeline.py` in OpenAI function-calling format (`{"type": "function", "function": {...}}`)
- **Tool dispatcher:** `run_tool()` maps tool call names from the agent to the corresponding Python functions
- **Data profile:** `get_data_profile()` builds the column summary passed to the agent — adding new fields here gives the agent more signal to make better decisions
- **Pre-cleaning:** `transform()` runs lowercasing, `drop_duplicates`, and `replace_nan_strings` before calling `run_agent()` — these are deterministic and not delegated to the model
