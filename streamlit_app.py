import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Spending Analyzer", layout="wide")


REQUIRED_COLUMNS = ["Transaction Date", "Amount", "Code", "Details"]
DEFAULT_CATEGORY = "Uncategorized"


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")

    return normalise_columns(df)


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Expected: "
            + ", ".join(REQUIRED_COLUMNS)
        )


def prepare_spending_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    validate_required_columns(df)

    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    for col in ["Code", "Details"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    spending_df = df[df["Amount"] < 0].copy()
    spending_df = spending_df[spending_df["Code"].str.casefold() != "transfer"].copy()

    spending_df["Spending Value"] = spending_df["Amount"].abs()
    spending_df["Vendor Name"] = spending_df["Code"]

    special_code_mask = spending_df["Code"].str.replace(r"\s+", " ", regex=True).str.strip().isin(["", "0991 C"])
    spending_df.loc[special_code_mask, "Vendor Name"] = spending_df.loc[special_code_mask, "Details"]

    spending_df["Vendor Name"] = spending_df["Vendor Name"].fillna("").astype(str).str.strip()
    spending_df.loc[spending_df["Vendor Name"].eq(""), "Vendor Name"] = "Unknown Vendor Name"

    spending_df["Date"] = spending_df["Transaction Date"].dt.normalize()
    spending_df["Month"] = spending_df["Transaction Date"].dt.to_period("M").astype(str)
    spending_df["Week Start"] = spending_df["Transaction Date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    return spending_df


def apply_category_mapping(spending_df: pd.DataFrame, vendor_category_map: dict) -> pd.DataFrame:
    df = spending_df.copy()
    df["Category"] = df["Vendor Name"].map(vendor_category_map).fillna(DEFAULT_CATEGORY)
    return df


def vendor_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Category", "Vendor Name"], dropna=False)["Spending Value"]
        .sum()
        .reset_index()
        .sort_values("Spending Value", ascending=False)
        .reset_index(drop=True)
    )
    total = summary["Spending Value"].sum()
    if total > 0:
        summary["Percent of Total Spending"] = summary["Spending Value"] / total * 100
    else:
        summary["Percent of Total Spending"] = 0.0
    return summary


def weekly_spending_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Week Start", dropna=False)["Spending Value"]
        .sum()
        .reset_index()
        .sort_values("Week Start")
        .reset_index(drop=True)
    )


def make_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("Spending Analyzer")
st.write("Upload your bank transaction file, assign categories to vendors, and view your spending instantly.")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload a transaction file to begin.")
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded_file)
    spending_df = prepare_spending_data(raw_df)
except Exception as exc:
    st.error(f"Error reading file: {exc}")
    st.stop()

st.success("File uploaded and processed successfully.")

st.subheader("Quick Check")
col1, col2, col3 = st.columns(3)
col1.metric("Raw Rows", f"{len(raw_df):,}")
col2.metric("Spending Rows", f"{len(spending_df):,}")
col3.metric("Unique Vendors", f"{spending_df['Vendor Name'].nunique():,}")

unique_vendors = sorted(spending_df["Vendor Name"].dropna().unique().tolist())

st.subheader("Assign Categories")
st.caption("Set a category for each vendor. Leave blank to keep it as Uncategorized.")

preset_categories_text = st.text_input(
    "Preset categories (comma separated)",
    value="Groceries, Eating Out, Transport, Bills, Health, Entertainment, Shopping, Vape, Cafe, Other"
)
preset_categories = [x.strip() for x in preset_categories_text.split(",") if x.strip()]
category_options = [""] + preset_categories

if "vendor_category_map" not in st.session_state:
    st.session_state.vendor_category_map = {}

with st.expander("Vendor category editor", expanded=True):
    for vendor in unique_vendors:
        current_value = st.session_state.vendor_category_map.get(vendor, "")
        selected = st.selectbox(
            f"{vendor}",
            options=category_options,
            index=category_options.index(current_value) if current_value in category_options else 0,
            key=f"vendor_{vendor}"
        )
        if selected == "":
            st.session_state.vendor_category_map.pop(vendor, None)
        else:
            st.session_state.vendor_category_map[vendor] = selected

categorized_df = apply_category_mapping(spending_df, st.session_state.vendor_category_map)

summary_df = vendor_summary_table(categorized_df)
weekly_df = weekly_spending_table(categorized_df)

st.subheader("Result A: Vendor Summary Table")
display_summary = summary_df.copy()
display_summary["Spending Value"] = display_summary["Spending Value"].round(2)
display_summary["Percent of Total Spending"] = display_summary["Percent of Total Spending"].round(2)
st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.subheader("Result B: Weekly Spending Plot")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(weekly_df["Week Start"], weekly_df["Spending Value"])
ax.set_xlabel("Week Start")
ax.set_ylabel("Spending")
ax.set_title("Weekly Spending")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Download Organized Spending")
download_df = categorized_df[
    ["Transaction Date", "Amount", "Spending Value", "Vendor Name", "Category", "Code", "Details"]
].copy()

st.download_button(
    label="Download organized spending CSV",
    data=make_download_csv(download_df),
    file_name="organized_spending.csv",
    mime="text/csv",
)

with st.expander("Preview organized transactions"):
    preview_df = download_df.copy()
    preview_df["Spending Value"] = preview_df["Spending Value"].round(2)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
