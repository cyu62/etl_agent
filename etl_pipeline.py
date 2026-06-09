import json
import logging
import pandas as pd
from datetime import datetime
from openai import OpenAI
import os

MODEL = "qwen2.5:7b"
MAX_ITERATIONS = 20
MAX_RETRIES = 3

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


os.makedirs("audit", exist_ok=True)
logging.basicConfig(
    filename=f"audit/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

def audit(msg):
    logging.info(msg)
    print(msg)

# --- Your existing ETL functions (unchanged) ---
def extract(filepath):
    if isinstance(filepath, pd.DataFrame):
        return filepath
    if hasattr(filepath, "read"):
        return pd.read_csv(filepath)
    if filepath.endswith(".json"):
        return pd.read_json(filepath)
    return pd.read_csv(filepath)

def transform(data, duck_db_path=None, duck_table=None):
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.lower()
    data, _ = drop_duplicates(data)
    data, _ = replace_nan_strings(data)
    data = run_agent(data, duck_db_path=duck_db_path, duck_table=duck_table)
    # Deterministically enforce target schema column names after the agent runs
    if duck_db_path and duck_table:
        target_schema = get_duckdb_schema(duck_db_path, duck_table)
        if target_schema:
            data = enforce_schema_columns(data, target_schema)
    return data

def load(data, filepath="cleaned_output.csv"):
    data.to_csv(filepath, index=False)
    return filepath

def load_to_duckdb(df, db_path, table, replace=False):
    import duckdb
    conn = duckdb.connect(db_path)
    try:
        table_exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0] > 0

        if table_exists and not replace:
            existing_cols = {row[0] for row in conn.execute(f"DESCRIBE {table}").fetchall()}
            incoming_cols = set(df.columns)
            missing = existing_cols - incoming_cols
            extra = incoming_cols - existing_cols
            if missing or extra:
                raise ValueError(
                    f"Schema mismatch — cannot append.\n"
                    f"  Columns in table but not in new data: {sorted(missing) or 'none'}\n"
                    f"  Columns in new data but not in table: {sorted(extra) or 'none'}\n"
                    f"Check 'Replace existing table' to overwrite instead."
                )
            conn.execute(f"INSERT INTO {table} SELECT * FROM df")
            mode = "appended"
        else:
            conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
            mode = "replaced" if (replace and table_exists) else "created"

        nrows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        audit(f"DuckDB load: {mode} table '{table}' in {db_path} — total rows now {nrows}")
        return nrows, mode
    finally:
        conn.close()



def enforce_schema_columns(df, target_schema):
    target_cols = list(target_schema.keys())
    incoming_cols = list(df.columns)

    # Nothing to do if columns already match exactly
    if incoming_cols == target_cols:
        return df

    # Build a rename map: for each target column not already present,
    # find the best matching incoming column by stripping and lowercasing both sides
    rename_map = {}
    unmatched_target = [c for c in target_cols if c not in incoming_cols]
    unmatched_incoming = [c for c in incoming_cols if c not in target_cols]

    for target_col in unmatched_target:
        target_norm = target_col.lower().replace(" ", "_").replace("-", "_")
        best = None
        for inc_col in unmatched_incoming:
            inc_norm = inc_col.lower().replace(" ", "_").replace("-", "_")
            if inc_norm == target_norm:
                best = inc_col
                break
        # Fallback: align by position if counts match and no fuzzy match found
        if best is None and len(unmatched_target) == len(unmatched_incoming):
            idx = unmatched_target.index(target_col)
            best = unmatched_incoming[idx]
        if best:
            rename_map[best] = target_col
            unmatched_incoming.remove(best)

    if rename_map:
        audit(f"Schema enforcement: renaming columns {rename_map}")
        df = df.rename(columns=rename_map)

    # Reorder to match target schema, keeping any extra columns at the end
    ordered = [c for c in target_cols if c in df.columns]
    extras = [c for c in df.columns if c not in target_cols]
    df = df[ordered + extras]

    return df

def get_duckdb_schema(db_path, table):
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        try:
            exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
            ).fetchone()[0] > 0
            if not exists:
                return None
            rows = conn.execute(f"DESCRIBE {table}").fetchall()
            return {row[0]: row[1] for row in rows}
        finally:
            conn.close()
    except Exception:
        return None

def get_data_profile(df, target_schema=None):
    def col_profile(series):
        profile = {
            "dtype": str(series.dtype),
            "null_pct": round(series.isna().mean(), 3),
            "sample_values": series.dropna().head(5).tolist(),
        }
        if series.dtype == object:
            non_null = series.dropna().astype(str)
            has_upper = non_null.str.isupper().any()
            has_lower = non_null.str.islower().any()
            has_mixed = non_null.apply(lambda x: not x.isupper() and not x.islower() and x.isalpha()).any()
            profile["mixed_case"] = bool((has_upper and has_lower) or has_mixed)
            profile["unique_count"] = int(non_null.nunique())
        if pd.api.types.is_numeric_dtype(series):
            profile["has_negatives"] = bool((series.dropna() < 0).any())
        return profile

    profile = {
        "shape": df.shape,
        "columns": {col: col_profile(df[col]) for col in df.columns},
    }
    if target_schema:
        profile["target_table_schema"] = target_schema
    return profile

def drop_column(df, column):
    df = df.drop(columns=column)
    return df, f"Dropped column '{column}'."

def fill_nulls(df, column, strategy):
    null_count = df[column].isna().sum()
    if strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(strategy)
    return df, f"Filled {null_count} null(s) in '{column}' using strategy '{strategy}'."

def convert_dtype(df, column, dtype):
    before = str(df[column].dtype)
    df[column] = df[column].astype(dtype)
    return df, f"Converted '{column}' from {before} to {dtype}."

def rename_column(df, old_name, new_name):
    df = df.rename(columns={old_name: new_name})
    return df, f"Renamed column '{old_name}' to '{new_name}'."

def strip_characters(df, column, chars):
    before = df[column].astype(str).copy()
    df[column] = df[column].astype(str).str.strip(chars)
    changed = (before != df[column].astype(str)).sum()
    return df, f"Stripped '{chars}' from '{column}'; {changed} value(s) changed."

def standardize_case(df, column, case):
    before = df[column].astype(str).copy()
    if case == "lower":
        df[column] = df[column].astype(str).str.lower()
    elif case == "upper":
        df[column] = df[column].astype(str).str.upper()
    changed = (before != df[column].astype(str)).sum()
    return df, f"Standardized '{column}' to {case}case; {changed} value(s) changed."

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df)
    return df, f"Removed {removed} duplicate row(s); {len(df)} rows remain."

def standardize_date(df, column):
    import re
    parsed, failed = 0, 0
    def parse_date(val):
        nonlocal parsed, failed
        if pd.isna(val):
            return val
        val = str(val).strip()
        m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', val)
        if m:
            part1, part2, year = m.groups()
            if int(part1) > 12:
                parsed += 1
                return f"{year}-{part2.zfill(2)}-{part1.zfill(2)}"
            parsed += 1
            return f"{year}-{part1.zfill(2)}-{part2.zfill(2)}"
        m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', val)
        if m:
            parsed += 1
            return val
        failed += 1
        return val
    df[column] = df[column].apply(parse_date)
    return df, f"Standardized dates in '{column}': {parsed} parsed, {failed} could not be parsed."

def standardize_phone(df, column):
    import re
    formatted, failed = 0, 0
    def clean_phone(val):
        nonlocal formatted, failed
        if pd.isna(val):
            return val
        digits = re.sub(r'\D', '', str(val))
        if len(digits) == 11 and digits[0] == '1':
            digits = digits[1:]
        if len(digits) == 10:
            formatted += 1
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        failed += 1
        return str(val)
    df[column] = df[column].apply(clean_phone)
    return df, f"Standardized phones in '{column}': {formatted} formatted, {failed} could not be parsed."

def validate_email(df, column):
    import re
    pattern = re.compile(r'^[^@]+@[^@]+\.[^@]+$')
    valid, invalid = 0, 0
    def check_email(x):
        nonlocal valid, invalid
        if pd.notna(x) and pattern.match(str(x)):
            valid += 1
            return x
        invalid += 1
        return ""
    df[column] = df[column].apply(check_email)
    return df, f"Validated emails in '{column}': {valid} valid, {invalid} invalid (cleared)."

def replace_nan_strings(df):
    before = (df == "nan").sum().sum() + (df == "NaN").sum().sum()
    df = df.replace("nan", "", regex=False)
    df = df.replace("NaN", "", regex=False)
    return df, f"Replaced {before} literal 'nan'/'NaN' string(s) with empty string."

def flag_negative_balance(df, column):
    flagged = df[column].apply(lambda x: pd.notna(x) and x < 0).sum()
    df["balance_flag"] = df[column].apply(
        lambda x: "negative_balance" if pd.notna(x) and x < 0 else ""
    )
    return df, f"Flagged {flagged} negative value(s) in '{column}' — added 'balance_flag' column."

def trim_whitespace(df, column):
    before = df[column].astype(str).copy()
    df[column] = df[column].astype(str).str.strip()
    changed = (before != df[column].astype(str)).sum()
    return df, f"Trimmed whitespace in '{column}'; {changed} value(s) changed."


# --- Tool definitions (OpenAI format for Ollama) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "drop_column",
            "description": "Drop a column from the dataframe",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name to drop"}
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_dtype",
            "description": "Convert a column to a different data type",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"},
                    "dtype": {"type": "string", "description": "Target type: float, int, str"},
                },
                "required": ["column", "dtype"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_column",
            "description": "Rename a column",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current column name"},
                    "new_name": {"type": "string", "description": "New column name"},
                },
                "required": ["old_name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "strip_characters",
            "description": "Strip leading/trailing characters from a column, e.g. '$,' to clean currency values",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"},
                    "chars": {"type": "string", "description": "Characters to strip"},
                },
                "required": ["column", "chars"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "standardize_case",
            "description": "Convert all string values in a column to uppercase or lowercase for consistency",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"},
                    "case": {"type": "string", "description": "Either 'lower' or 'upper'"},
                },
                "required": ["column", "case"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drop_duplicates",
            "description": "Remove duplicate rows where all column values are identical, keeping the first occurrence",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "standardize_date",
            "description": "Standardize all date values in a column to YYYY-MM-DD format",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name containing dates"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "standardize_phone",
            "description": "Standardize phone numbers to XXX-XXX-XXXX format, removing country codes and special characters",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name containing phone numbers"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_email",
            "description": "Validate emails in a column — replace invalid ones (missing domain, no dot after @) with empty string",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name containing email addresses"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_nan_strings",
            "description": "Replace any literal 'nan' or 'NaN' strings across all columns with empty string",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_negative_balance",
            "description": "Add a balance_flag column marking rows where the balance column is negative",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name containing balance values"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trim_whitespace",
            "description": "Strip leading and trailing whitespace from all string values in a column",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this when cleaning is complete",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# --- Tool dispatcher ---
def run_tool(df, name, tool_input):
    try:
        if name == "drop_column":
            return drop_column(df, tool_input["column"])
        elif name == "convert_dtype":
            return convert_dtype(df, tool_input["column"], tool_input["dtype"])
        elif name == "rename_column":
            return rename_column(df, tool_input["old_name"], tool_input["new_name"])
        elif name == "strip_characters":
            return strip_characters(df, tool_input["column"], tool_input["chars"])
        elif name == "standardize_case":
            return standardize_case(df, tool_input["column"], tool_input["case"])
        elif name == "drop_duplicates":
            return drop_duplicates(df)
        elif name == "standardize_date":
            return standardize_date(df, tool_input["column"])
        elif name == "standardize_phone":
            return standardize_phone(df, tool_input["column"])
        elif name == "validate_email":
            return validate_email(df, tool_input["column"])
        elif name == "replace_nan_strings":
            return replace_nan_strings(df)
        elif name == "flag_negative_balance":
            return flag_negative_balance(df, tool_input["column"])
        elif name == "trim_whitespace":
            return trim_whitespace(df, tool_input["column"])
        else:
            return df, f"Unknown tool: {name}"
    except Exception as e:
        return df, f"Error: {e}"

# --- Agent loop ---
def run_agent(source, duck_db_path=None, duck_table=None):
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, (list, dict)):
        df = pd.DataFrame(source)
    elif source.endswith(".json"):
        df = pd.read_json(source)
    else:
        df = pd.read_csv(source)

    target_schema = None
    if duck_db_path and duck_table:
        target_schema = get_duckdb_schema(duck_db_path, duck_table)
        if target_schema:
            audit(f"Target DuckDB table '{duck_table}' exists — schema loaded for agent reconciliation.")

    profile = get_data_profile(df, target_schema=target_schema)
    audit(f"Starting agent. Dataset shape: {df.shape}")

    system_prompt = (
        "You are a data cleaning agent. Your job is to analyze a data profile and call tools to clean the data.\n\n"
        "Follow this process:\n"
        "1. Review every column in the profile — dtype, null percentage, casing, sample values.\n"
        "2. For each issue you find, call the appropriate tool to fix it.\n"
        "3. Work through ALL columns before calling finish — do not stop after the first one.\n\n"
        "Rules:\n"
        "- Always call drop_duplicates and replace_nan_strings at the start.\n"
        "- Call trim_whitespace on any string column that may have leading/trailing spaces.\n"
        "- NEVER fill null or missing values with placeholder text like 'unknown', 'N/A', 'none', or 'null'. Leave missing cells empty.\n"
        "- NEVER impute, estimate, or fill missing values in numeric columns. Do not use mean, median, or mode to fill nulls. Leave them empty.\n"
        "- If a string column contains values like 'unknown', 'N/A', 'none', or 'null', replace them with empty string using replace_nan_strings.\n"
        "- Strip currency symbols ('$', ',') or units before converting to a numeric dtype.\n"
        "- Convert columns that look numeric but have dtype 'object' to float or int.\n"
        "- Rename columns with spaces or special characters to use underscores.\n"
        "- For any column whose name contains 'date', 'dob', 'time', 'created', 'updated', 'timestamp', or '_at', or whose sample values look like dates: call standardize_date on it. Apply this to EVERY date column found, not just the first.\n"
        "- When standardizing dates, inspect each value individually. If the number in the day position is greater than 12, treat it as DD/MM/YYYY and reorder accordingly to produce YYYY-MM-DD.\n"
        "- If a column name contains 'phone', call standardize_phone on it.\n"
        "- If a column name contains 'email', call validate_email on it.\n"
        "- If a numeric column has has_negatives: true in its profile, call flag_negative_balance on it.\n"
        "- If a string column has mixed_case: true, call standardize_case with 'lower' to normalize it.\n"
        "- If the profile contains a 'target_table_schema' key, it means the data will be appended to an existing table. "
        "You MUST ensure the incoming columns match the target schema exactly before calling finish. "
        "For each column in target_table_schema: if the name is missing from the incoming data, check if a differently-named column contains the same data and rename it. "
        "If a column's dtype does not match, call convert_dtype to align it. "
        "Do not finish until every column in the incoming data matches a column in the target schema by name and type.\n"
        "- Only call finish when every column has been reviewed and all issues are resolved."
    )

    user_prompt = (
        f"Here is the profile of the dataset:\n{json.dumps(profile, indent=2)}\n\n"
        "Go through each column, identify issues, and fix them using the available tools. "
        "State which issue you are fixing and why before each tool call. "
        "When all columns are clean, call the finish tool."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    iteration = 0
    retries = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        audit(f"--- Iteration {iteration} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(msg)

        if msg.content:
            audit(f"Agent: {msg.content}")

        # Agent stopped without calling finish — retry with a nudge
        if finish_reason == "stop" or not msg.tool_calls:
            retries += 1
            if retries > MAX_RETRIES:
                audit("Max retries reached. Stopping.")
                break
            audit(f"Agent stopped early (retry {retries}/{MAX_RETRIES}). Nudging...")
            messages.append({
                "role": "user",
                "content": "You have not called the finish tool yet. Continue reviewing the remaining columns and call finish when all are clean.",
            })
            continue

        done = False
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name == "finish":
                done = True
                result_msg = "Cleaning complete."
                audit("Agent called finish — cleaning complete.")
            else:
                audit(f"Tool call: {name}({args})")
                df, result_msg = run_tool(df, name, args)
                audit(f"Tool result: {result_msg}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_msg,
            })

        if done:
            break

    else:
        audit(f"Hit max iteration limit ({MAX_ITERATIONS}). Returning current state.")

    audit(f"Agent finished. Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "bank_customers_raw.csv"
    data = extract(filepath)
    cleaned = transform(data)
    load(cleaned)
