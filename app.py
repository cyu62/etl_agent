import os
import glob
import duckdb
import streamlit as st
from etl_pipeline import extract, transform, load, load_to_duckdb

DUCKDB_DIR = "data"
os.makedirs(DUCKDB_DIR, exist_ok=True)

st.title("ETL Pipeline — CSV Cleaner")
st.write("Upload a CSV file and the agent will clean it automatically.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = extract(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head(20))
    st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns")

    st.subheader("DuckDB Destination (optional)")
    st.caption("Leave blank to skip. Files are stored in the data/ folder.")
    col1, col2 = st.columns(2)
    with col1:
        duck_db_name = st.text_input("DuckDB file name", placeholder="customers.duckdb")
    with col2:
        duck_table = st.text_input("Table name", placeholder="cleaned_data")

    duck_db_path = os.path.join(DUCKDB_DIR, duck_db_name) if duck_db_name else ""
    load_to_duck = bool(duck_db_name and duck_table)

    if st.button("Run Pipeline", type="primary"):
        with st.status("Cleaning data...", expanded=True) as status:
            try:
                st.write("Extracting...")
                uploaded_file.seek(0)
                data = extract(uploaded_file)

                st.write("Running AI agent...")
                # Only pass DuckDB params if the file already exists — schema
                # reconciliation only applies when appending to an existing table.
                # Passing a path to a non-existent file would cause DuckDB to create it here.
                db_exists = os.path.exists(duck_db_path) if load_to_duck else False
                cleaned = transform(
                    data,
                    duck_db_path=duck_db_path if db_exists else None,
                    duck_table=duck_table if db_exists else None,
                )

                st.write("Saving cleaned CSV...")
                output_filename = f"cleaned_{uploaded_file.name}"
                load(cleaned, output_filename)

                status.update(label="Done!", state="complete")

                st.session_state["cleaned_df"] = cleaned
                st.session_state["output_filename"] = output_filename
                st.session_state["duck_db_path"] = duck_db_path
                st.session_state["duck_table"] = duck_table
                st.session_state["load_to_duck"] = load_to_duck
                st.session_state["raw_shape"] = df.shape

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Pipeline failed: {e}")

if "cleaned_df" in st.session_state:
    cleaned = st.session_state["cleaned_df"]
    output_filename = st.session_state["output_filename"]
    raw_shape = st.session_state["raw_shape"]

    st.subheader("Cleaned Data")
    st.dataframe(cleaned.head(20))
    rows_removed = raw_shape[0] - cleaned.shape[0]
    cols_added = cleaned.shape[1] - raw_shape[1]
    st.caption(f"{cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    st.info(f"Removed {rows_removed} duplicate rows. Added {cols_added} new column(s).")

    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cleaned CSV",
        data=csv_bytes,
        file_name=output_filename,
        mime="text/csv",
    )

    if st.session_state.get("load_to_duck"):
        st.divider()
        duck_db_path = st.session_state["duck_db_path"]
        duck_table = st.session_state["duck_table"]
        st.write(f"Ready to load into **{duck_db_path}** → table **{duck_table}**")

        replace_table = st.checkbox("Replace existing table (overwrites all current data)")

        if st.button("Load into DuckDB", type="primary"):
            with st.spinner("Loading into DuckDB..."):
                try:
                    nrows, mode = load_to_duckdb(cleaned, duck_db_path, duck_table, replace=replace_table)
                    if mode == "appended":
                        st.success(f"Appended to existing table '{duck_table}' — {nrows} total rows now in {duck_db_path}")
                    elif mode == "replaced":
                        st.success(f"Replaced table '{duck_table}' with {nrows} new rows in {duck_db_path}")
                    else:
                        st.success(f"Created table '{duck_table}' with {nrows} rows in {duck_db_path}")
                except Exception as e:
                    st.error(f"DuckDB load failed: {e}")

# --- DuckDB Browser ---
st.divider()
st.subheader("DuckDB Browser")

db_files = glob.glob(os.path.join(DUCKDB_DIR, "*.duckdb"))

if not db_files:
    st.caption("No .duckdb files found in the data/ folder yet.")
else:
    db_names = [os.path.basename(f) for f in db_files]
    selected_db = st.selectbox("Select a database", db_names)
    selected_db_path = os.path.join(DUCKDB_DIR, selected_db)

    try:
        conn = duckdb.connect(selected_db_path, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        if not tables:
            st.caption("No tables found in this database.")
        else:
            selected_table = st.selectbox("Select a table", tables)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {selected_table}").fetchone()[0]
            st.caption(f"{row_count} rows in {selected_table}")

            preview_df = conn.execute(f"SELECT * FROM {selected_table} LIMIT 500").df()
            st.dataframe(preview_df)

            csv_bytes = preview_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {selected_table} as CSV",
                data=csv_bytes,
                file_name=f"{selected_table}.csv",
                mime="text/csv",
            )
        conn.close()
    except Exception as e:
        st.error(f"Could not open database: {e}")
