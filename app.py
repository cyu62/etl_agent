import streamlit as st
from etl_pipeline import extract, transform, load

st.title("ETL Pipeline — CSV Cleaner")
st.write("Upload a CSV file and the agent will clean it automatically.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = extract(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head(20))
    st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns")

    if st.button("Run Pipeline", type="primary"):
        with st.status("Cleaning data...", expanded=True) as status:
            try:
                st.write("Extracting...")
                uploaded_file.seek(0)
                data = extract(uploaded_file)

                st.write("Running AI agent...")
                cleaned = transform(data)

                st.write("Loading...")
                output_filename = f"cleaned_{uploaded_file.name}"
                load(cleaned, output_filename)

                status.update(label="Done!", state="complete")

                st.subheader("Cleaned Data")
                st.dataframe(cleaned.head(20))
                rows_removed = df.shape[0] - cleaned.shape[0]
                cols_added = cleaned.shape[1] - df.shape[1]
                st.caption(f"{cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
                st.info(f"Removed {rows_removed} duplicate rows. Added {cols_added} new column(s).")

                csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Cleaned CSV",
                    data=csv_bytes,
                    file_name=output_filename,
                    mime="text/csv",
                )

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Pipeline failed: {e}")
