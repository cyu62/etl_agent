import numpy as np
import pandas as pd
import pytest

from etl_pipeline import (
    convert_dtype,
    drop_column,
    drop_duplicates,
    enforce_schema_columns,
    flag_negative_balance,
    get_duckdb_schema,
    load_to_duckdb,
    rename_column,
    replace_nan_strings,
    run_tool,
    standardize_case,
    standardize_date,
    standardize_phone,
    strip_characters,
    trim_whitespace,
    validate_email,
    validate_table_name,
)


# --- replace_nan_strings ---

def test_replace_nan_strings_covers_all_placeholders():
    df = pd.DataFrame({
        "a": ["nan", "N/A", " UNKNOWN ", "None", "null", "keep"],
        "b": [1, 2, 3, 4, 5, 6],
    })
    df, msg = replace_nan_strings(df)
    assert df["a"].tolist() == ["", "", "", "", "", "keep"]
    assert df["b"].tolist() == [1, 2, 3, 4, 5, 6]
    assert "5" in msg


def test_replace_nan_strings_leaves_real_nan_and_substrings_alone():
    df = pd.DataFrame({"a": [np.nan, "nantucket", "nonebut", "banana"]})
    df, _ = replace_nan_strings(df)
    assert pd.isna(df["a"].iloc[0])
    assert df["a"].tolist()[1:] == ["nantucket", "nonebut", "banana"]


# --- basic column tools ---

def test_drop_column():
    df = pd.DataFrame({"a": [1], "b": [2]})
    df, _ = drop_column(df, "b")
    assert list(df.columns) == ["a"]


def test_rename_column():
    df = pd.DataFrame({"old": [1]})
    df, _ = rename_column(df, "old", "new")
    assert list(df.columns) == ["new"]


def test_convert_dtype_object_to_float():
    df = pd.DataFrame({"x": ["1.5", "2.0"]})
    df, _ = convert_dtype(df, "x", "float")
    assert df["x"].dtype == float
    assert df["x"].tolist() == [1.5, 2.0]


def test_strip_characters():
    df = pd.DataFrame({"price": ["$100", "$2,000,"]})
    df, _ = strip_characters(df, "price", "$,")
    assert df["price"].tolist() == ["100", "2,000"]


def test_standardize_case_upper():
    df = pd.DataFrame({"s": ["Mixed", "case"]})
    df, _ = standardize_case(df, "s", "upper")
    assert df["s"].tolist() == ["MIXED", "CASE"]


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    df, msg = drop_duplicates(df)
    assert len(df) == 2
    assert "1 duplicate" in msg


def test_trim_whitespace():
    df = pd.DataFrame({"s": ["  a ", "b"]})
    df, _ = trim_whitespace(df, "s")
    assert df["s"].tolist() == ["a", "b"]


# --- format standardizers ---

def test_standardize_date_formats():
    df = pd.DataFrame({"d": ["12/31/2024", "31/12/2024", "2024-01-05", "not a date", np.nan]})
    df, _ = standardize_date(df, "d")
    assert df["d"].tolist()[:3] == ["2024-12-31", "2024-12-31", "2024-01-05"]
    assert df["d"].iloc[3] == "not a date"
    assert pd.isna(df["d"].iloc[4])


def test_standardize_phone():
    df = pd.DataFrame({"p": ["(555) 123-4567", "1-555-123-4567", "12345"]})
    df, _ = standardize_phone(df, "p")
    assert df["p"].tolist() == ["555-123-4567", "555-123-4567", "12345"]


def test_validate_email():
    df = pd.DataFrame({"e": ["a@b.com", "bad-email", np.nan]})
    df, _ = validate_email(df, "e")
    assert df["e"].tolist() == ["a@b.com", "", ""]


def test_flag_negative_balance():
    df = pd.DataFrame({"bal": [-5.0, 10.0, np.nan]})
    df, _ = flag_negative_balance(df, "bal")
    assert df["bal_negative_flag"].tolist() == ["negative", "", ""]


def test_flag_negative_balance_two_columns_dont_collide():
    df = pd.DataFrame({"balance": [-1.0, 2.0], "income": [3.0, -4.0]})
    df, _ = flag_negative_balance(df, "balance")
    df, _ = flag_negative_balance(df, "income")
    assert df["balance_negative_flag"].tolist() == ["negative", ""]
    assert df["income_negative_flag"].tolist() == ["", "negative"]


# --- dispatcher ---

def test_run_tool_unknown_tool_returns_message():
    df = pd.DataFrame({"a": [1]})
    out, msg = run_tool(df, "no_such_tool", {})
    assert msg == "Unknown tool: no_such_tool"
    assert out is df


def test_run_tool_error_is_reported_not_raised():
    df = pd.DataFrame({"x": ["abc"]})
    out, msg = run_tool(df, "convert_dtype", {"column": "x", "dtype": "int"})
    assert msg.startswith("Error:")
    assert out["x"].tolist() == ["abc"]


def test_run_tool_does_not_mutate_input():
    df = pd.DataFrame({"s": ["  a "]})
    out, _ = run_tool(df, "trim_whitespace", {"column": "s"})
    assert df["s"].tolist() == ["  a "]  # caller's frame untouched
    assert out["s"].tolist() == ["a"]


def test_transform_does_not_mutate_caller(monkeypatch):
    import etl_pipeline
    monkeypatch.setattr(etl_pipeline, "run_agent", lambda df, **kw: df)
    raw = pd.DataFrame({"s": ["Keep Case", "Keep Case"]})
    out = etl_pipeline.transform(raw)
    assert len(raw) == 2  # caller's frame keeps its duplicate row
    assert len(out) == 1
    assert out["s"].tolist() == ["Keep Case"]  # no blanket lowercasing


# --- schema enforcement ---

TARGET = {"customer_id": "INT", "first_name": "VARCHAR", "balance": "FLOAT"}


def test_enforce_schema_exact_match_is_noop():
    df = pd.DataFrame(columns=["customer_id", "first_name", "balance"])
    out = enforce_schema_columns(df, TARGET)
    assert list(out.columns) == ["customer_id", "first_name", "balance"]


def test_enforce_schema_fuzzy_rename_and_reorder():
    df = pd.DataFrame(columns=["First Name", "customer-id", "balance"])
    out = enforce_schema_columns(df, TARGET)
    assert list(out.columns) == ["customer_id", "first_name", "balance"]


def test_enforce_schema_single_leftover_pair_is_renamed():
    df = pd.DataFrame(columns=["customer_id", "first_name", "bal_amt"])
    out = enforce_schema_columns(df, TARGET)
    assert list(out.columns) == ["customer_id", "first_name", "balance"]


def test_enforce_schema_refuses_ambiguous_positional_rename():
    df = pd.DataFrame(columns=["customer_id", "fname", "bal_amt"])
    out = enforce_schema_columns(df, TARGET)
    # Two unmatched on each side: no guessing — originals kept as extras at the end
    assert list(out.columns) == ["customer_id", "fname", "bal_amt"]


# --- agent loop ---

def test_agent_survives_malformed_tool_arguments(monkeypatch):
    from types import SimpleNamespace
    import etl_pipeline

    responses = [
        # First turn: model emits truncated JSON arguments
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="trim_whitespace", arguments='{"column": '),
        ),
        # Second turn: model recovers and finishes
        SimpleNamespace(id="call_2", function=SimpleNamespace(name="finish", arguments="")),
    ]
    seen_messages = []

    def fake_create(model, messages, tools, **kwargs):
        seen_messages.append(list(messages))
        tc = responses[len(seen_messages) - 1]
        msg = SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(etl_pipeline, "client", fake_client)

    df = pd.DataFrame({"a": [" x "]})
    out = etl_pipeline.run_agent(df)

    # No crash, the malformed call was not dispatched, and the agent got a
    # tool-role error message it could react to on the next turn
    assert out["a"].tolist() == [" x "]
    error_reply = seen_messages[1][-1]
    assert error_reply["role"] == "tool"
    assert error_reply["tool_call_id"] == "call_1"
    assert "not valid JSON" in error_reply["content"]


# --- DuckDB table name validation and load ---

@pytest.mark.parametrize("bad", ["", "bad name", "x; DROP TABLE t", "1table", "t-1"])
def test_validate_table_name_rejects(bad):
    with pytest.raises(ValueError):
        validate_table_name(bad)


def test_validate_table_name_accepts():
    validate_table_name("customers_2024")
    validate_table_name("_tmp")


def test_load_to_duckdb_create_append_and_mismatch(tmp_path):
    db = str(tmp_path / "t.duckdb")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    nrows, mode = load_to_duckdb(df, db, "t1")
    assert (nrows, mode) == (2, "created")

    nrows, mode = load_to_duckdb(df, db, "t1")
    assert (nrows, mode) == (4, "appended")

    with pytest.raises(ValueError, match="Schema mismatch"):
        load_to_duckdb(pd.DataFrame({"a": [1], "c": [2]}), db, "t1")

    nrows, mode = load_to_duckdb(df, db, "t1", replace=True)
    assert (nrows, mode) == (2, "replaced")

    assert get_duckdb_schema(db, "t1") is not None
    assert get_duckdb_schema(db, "missing") is None


def test_load_to_duckdb_rejects_injection(tmp_path):
    db = str(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="Invalid table name"):
        load_to_duckdb(pd.DataFrame({"a": [1]}), db, "t1; DROP TABLE x")
