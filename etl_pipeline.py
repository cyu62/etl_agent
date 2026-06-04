import json
import pandas as pd
import requests
from openai import OpenAI

MODEL = "qwen2.5:3b"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# --- Your existing ETL functions (unchanged) ---
def extract(filepath):
    if isinstance(filepath, pd.DataFrame):
        return filepath
    if hasattr(filepath, "read"):
        return pd.read_csv(filepath)
    if filepath.endswith(".json"):
        return pd.read_json(filepath)
    return pd.read_csv(filepath)

def transform(data):
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.lower()
    data = drop_duplicates(data)
    data = replace_nan_strings(data)
    return run_agent(data)

def load(data, filepath="cleaned_output.csv"):
    data.to_csv(filepath, index=False)
    return filepath



def get_data_profile(df):
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
        return profile

    return {
        "shape": df.shape,
        "columns": {col: col_profile(df[col]) for col in df.columns},
    }

def drop_column(df, column):

    df = df.drop(columns = column)
    return df

def fill_nulls(df, column, strategy):
    if strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(strategy)
    return df

def convert_dtype(df, column, dtype):
    df[column] = df[column].astype(dtype)
    return df

def rename_column(df, old_name, new_name):
    df = df.rename(columns={old_name: new_name})
    return df

def strip_characters(df, column, chars):
    df[column] = df[column].astype(str).str.strip(chars)
    return df

def standardize_case(df, column, case):
    if case == "lower":
        df[column] = df[column].astype(str).str.lower()
    elif case == "upper":
        df[column] = df[column].astype(str).str.upper()
    return df

def drop_duplicates(df):
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df

def standardize_date(df, column):
    import re
    def parse_date(val):
        if pd.isna(val):
            return val
        val = str(val).strip()
        # Try MM/DD/YYYY or MM-DD-YYYY
        m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', val)
        if m:
            month, day, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        # Try DD/MM/YYYY (ambiguous — only use if day > 12)
        m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', val)
        if m:
            part1, part2, year = m.groups()
            if int(part1) > 12:
                return f"{year}-{part2.zfill(2)}-{part1.zfill(2)}"
        # Already YYYY-MM-DD
        m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', val)
        if m:
            return val
        return val
    df[column] = df[column].apply(parse_date)
    return df

def standardize_phone(df, column):
    import re
    def clean_phone(val):
        if pd.isna(val):
            return val
        digits = re.sub(r'\D', '', str(val))
        if len(digits) == 11 and digits[0] == '1':
            digits = digits[1:]
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return str(val)
    df[column] = df[column].apply(clean_phone)
    return df

def validate_email(df, column):
    import re
    pattern = re.compile(r'^[^@]+@[^@]+\.[^@]+$')
    df[column] = df[column].apply(
        lambda x: x if pd.notna(x) and pattern.match(str(x)) else ""
    )
    return df

def replace_nan_strings(df):
    df = df.replace("nan", "", regex=False)
    df = df.replace("NaN", "", regex=False)
    return df

def flag_negative_balance(df, column):
    df["balance_flag"] = df[column].apply(
        lambda x: "negative_balance" if pd.notna(x) and x < 0 else ""
    )
    return df


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
            "name": "fill_nulls",
            "description": "Fill null values in a column. Strategy can be 'mean', 'median', 'mode', or a fixed value like '0' or 'unknown'",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"},
                    "strategy": {"type": "string", "description": "mean, median, mode, or a fixed fill value"},
                },
                "required": ["column", "strategy"],
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
            return drop_column(df, tool_input["column"]), "ok"
        elif name == "fill_nulls":
            return fill_nulls(df, tool_input["column"], tool_input["strategy"]), "ok"
        elif name == "convert_dtype":
            return convert_dtype(df, tool_input["column"], tool_input["dtype"]), "ok"
        elif name == "rename_column":
            return rename_column(df, tool_input["old_name"], tool_input["new_name"]), "ok"
        elif name == "strip_characters":
            return strip_characters(df, tool_input["column"], tool_input["chars"]), "ok"
        elif name == "standardize_case":
            return standardize_case(df, tool_input["column"], tool_input["case"]), "ok"
        elif name == "drop_duplicates":
            return drop_duplicates(df), "ok"
        elif name == "standardize_date":
            return standardize_date(df, tool_input["column"]), "ok"
        elif name == "standardize_phone":
            return standardize_phone(df, tool_input["column"]), "ok"
        elif name == "validate_email":
            return validate_email(df, tool_input["column"]), "ok"
        elif name == "replace_nan_strings":
            return replace_nan_strings(df), "ok"
        elif name == "flag_negative_balance":
            return flag_negative_balance(df, tool_input["column"]), "ok"
        else:
            return df, f"Unknown tool: {name}"
    except Exception as e:
        return df, f"Error: {e}"

# --- Agent loop ---
def run_agent(source):
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, (list, dict)):
        df = pd.DataFrame(source)
    elif source.endswith(".json"):
        df = pd.read_json(source)
    else:
        df = pd.read_csv(source)

    profile = get_data_profile(df)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data cleaning agent. Your job is to analyze a data profile and call tools to clean the data.\n\n"
                "Follow this process:\n"
                "1. Review every column in the profile — dtype, null percentage, casing, sample values.\n"
                "2. For each issue you find, call the appropriate tool to fix it.\n"
                "3. Work through all columns before calling finish.\n\n"
                "Rules:\n"
                "- Always call drop_duplicates and replace_nan_strings at the start.\n"
                "- Fill nulls in numeric columns with 'mean' or 'median'. Fill nulls in string columns with 'unknown'.\n"
                "- Strip currency symbols ('$', ',') or units before converting to a numeric dtype.\n"
                "- Convert columns that look numeric but have dtype 'object' to float or int.\n"
                "- Rename columns with spaces or special characters to use underscores.\n"
                "- If a column name contains 'date' or 'dob', call standardize_date on it.\n"
                "- If a column name contains 'phone', call standardize_phone on it.\n"
                "- If a column name contains 'email', call validate_email on it.\n"
                "- If a column name contains 'balance', call flag_negative_balance on it.\n"
                "- If a string column has mixed_case: true, call standardize_case with 'lower' to normalize it.\n"
                "- Only call finish when every column has been reviewed and all issues are resolved."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here is the profile of the dataset:\n{json.dumps(profile, indent=2)}\n\n"
                "Go through each column, identify issues, and fix them using the available tools. "
                "State which issue you are fixing and why before each tool call. "
                "When all columns are clean, call the finish tool."
            ),
        },
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message
        messages.append(msg)
        print(f"Agent response:\n{msg.content}\n")

        if response.choices[0].finish_reason == "stop" or not msg.tool_calls:
            print("Agent finished.")
            break

        done = False
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name == "finish":
                done = True
                result_msg = "Cleaning complete."
            else:
                print(f"Calling tool: {name}({args})")
                df, result_msg = run_tool(df, name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_msg,
            })

        if done:
            break

    return df

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "bank_customers_raw.csv"
    data = extract(filepath)
    cleaned = transform(data)
    load(cleaned)
