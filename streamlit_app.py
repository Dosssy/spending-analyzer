import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Spending Analyzer",
    layout="wide",
)

st.title("Spending Analyzer")
st.caption(
    "Upload your bank transaction file, prioritise the biggest vendors first, "
    "and stop when the remaining spending is too small to care about."
)


# =========================
# CONSTANTS
# =========================
REQUIRED_COLUMNS = ["Transaction Date", "Amount", "Code", "Details"]


# =========================
# HELPERS
# =========================
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

    special_code_mask = (
        spending_df["Code"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .isin(["", "0991 C"])
    )
    spending_df.loc[special_code_mask, "Vendor Name"] = spending_df.loc[special_code_mask, "Details"]

    spending_df["Vendor Name"] = spending_df["Vendor Name"].fillna("").astype(str).str.strip()
    spending_df.loc[spending_df["Vendor Name"].eq(""), "Vendor Name"] = "Unknown Vendor Name"

    spending_df["Date"] = spending_df["Transaction Date"].dt.normalize()
    spending_df["Month"] = spending_df["Transaction Date"].dt.to_period("M").astype(str)
    spending_df["Week Start"] = spending_df["Transaction Date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    return spending_df


def vendor_totals(spending_df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        spending_df.groupby("Vendor Name", dropna=False)["Spending Value"]
        .sum()
        .reset_index()
        .sort_values("Spending Value", ascending=False)
        .reset_index(drop=True)
    )
    return totals


def apply_category_mapping(
    spending_df: pd.DataFrame,
    vendor_category_map: dict,
    default_category: str,
    low_value_cutoff: float,
    low_value_category: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    totals_df = vendor_totals(spending_df)

    manual_vendors_df = totals_df[totals_df["Spending Value"] > low_value_cutoff].copy()
    auto_vendors_df = totals_df[totals_df["Spending Value"] <= low_value_cutoff].copy()

    final_map = {}

    for _, row in manual_vendors_df.iterrows():
        vendor = row["Vendor Name"]
        final_map[vendor] = vendor_category_map.get(vendor, default_category)

    for _, row in auto_vendors_df.iterrows():
        vendor = row["Vendor Name"]
        final_map[vendor] = low_value_category

    categorized_df = spending_df.copy()
    categorized_df["Category"] = categorized_df["Vendor Name"].map(final_map).fillna(default_category)

    return categorized_df, manual_vendors_df, auto_vendors_df


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


def category_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Category", dropna=False)["Spending Value"]
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


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


# =========================
# APP
# =========================
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

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

# =========================
# QUICK CHECK
# =========================
st.subheader("Quick Check")
qc1, qc2, qc3, qc4 = st.columns(4)
qc1.metric("Raw Rows", f"{len(raw_df):,}")
qc2.metric("Spending Rows", f"{len(spending_df):,}")
qc3.metric("Unique Vendors", f"{spending_df['Vendor Name'].nunique():,}")
qc4.metric("Total Spending", format_currency(spending_df["Spending Value"].sum()))

# =========================
# CONTROLS
# =========================
st.subheader("Category Controls")

control_col_1, control_col_2 = st.columns(2)

with control_col_1:
    preset_categories_text = st.text_input(
        "Preset categories (comma separated)",
        value="Groceries, Eating Out, Transport, Bills, Health, Entertainment, Shopping, Vape, Cafe, Miscellaneous, Other"
    )

    default_category = st.text_input(
        "Default category for untouched vendors",
        value="Uncategorised"
    )

with control_col_2:
    low_value_cutoff = st.number_input(
        "Auto-category cutoff amount",
        min_value=0.0,
        value=4.0,
        step=1.0,
        help="Any vendor with total spending at or below this amount will be automatically categorized."
    )

    low_value_category = st.text_input(
        "Automatic category for low-value vendors",
        value="Miscellaneous"
    )

preset_categories = [x.strip() for x in preset_categories_text.split(",") if x.strip()]

category_options = []
for category in [default_category] + preset_categories + [low_value_category]:
    if category not in category_options:
        category_options.append(category)

totals_df = vendor_totals(spending_df)
manual_queue_df = totals_df[totals_df["Spending Value"] > low_value_cutoff].copy()
auto_queue_df = totals_df[totals_df["Spending Value"] <= low_value_cutoff].copy()

if "vendor_category_map_v2" not in st.session_state:
    st.session_state.vendor_category_map_v2 = {}

# Clean out stale vendors if a new file is uploaded
current_vendor_set = set(manual_queue_df["Vendor Name"].tolist())
st.session_state.vendor_category_map_v2 = {
    k: v for k, v in st.session_state.vendor_category_map_v2.items()
    if k in current_vendor_set
}

# =========================
# PRIORITY QUEUE SUMMARY
# =========================
st.subheader("Vendor Prioritisation Queue")

pq1, pq2, pq3, pq4 = st.columns(4)
pq1.metric("Manual Vendors Remaining", f"{len(manual_queue_df):,}")
pq2.metric("Auto-Categorised Small Vendors", f"{len(auto_queue_df):,}")
pq3.metric("Spend in Manual Queue", format_currency(manual_queue_df["Spending Value"].sum()))
pq4.metric("Spend Auto-Sent to Tail Bucket", format_currency(auto_queue_df["Spending Value"].sum()))

st.caption(
    "Vendors are sorted from largest spending to smallest. "
    "Anything at or below the cutoff is automatically assigned to the low-value category."
)

# =========================
# CATEGORY ASSIGNMENT UI
# =========================
st.subheader("Assign Categories to Highest-Value Vendors First")

with st.expander("Manual vendor assignment queue", expanded=True):
    for _, row in manual_queue_df.iterrows():
        vendor = row["Vendor Name"]
        spend_amount = row["Spending Value"]
        current_value = st.session_state.vendor_category_map_v2.get(vendor, default_category)

        left, middle, right = st.columns([2, 5, 2])

        with left:
            selected = st.selectbox(
                "Category",
                options=category_options,
                index=category_options.index(current_value) if current_value in category_options else 0,
                key=f"vendor_category_{vendor}",
                label_visibility="collapsed"
            )

        with middle:
            st.markdown(f"**{vendor}**")

        with right:
            st.markdown(f"**{format_currency(spend_amount)}**")

        st.session_state.vendor_category_map_v2[vendor] = selected

if len(auto_queue_df) > 0:
    with st.expander("Auto-categorised low-value vendors", expanded=False):
        auto_display_df = auto_queue_df.copy()
        auto_display_df["Assigned Category"] = low_value_category
        auto_display_df["Spending Value"] = auto_display_df["Spending Value"].round(2)
        st.dataframe(auto_display_df, use_container_width=True, hide_index=True)

# =========================
# APPLY CATEGORIES
# =========================
categorized_df, manual_vendors_df, auto_vendors_df = apply_category_mapping(
    spending_df=spending_df,
    vendor_category_map=st.session_state.vendor_category_map_v2,
    default_category=default_category,
    low_value_cutoff=low_value_cutoff,
    low_value_category=low_value_category,
)

vendor_summary_df = vendor_summary_table(categorized_df)
category_summary_df = category_summary_table(categorized_df)
weekly_df = weekly_spending_table(categorized_df)

# =========================
# RESULTS
# =========================
st.subheader("Result A: Vendor Summary Table")
display_vendor_summary = vendor_summary_df.copy()
display_vendor_summary["Spending Value"] = display_vendor_summary["Spending Value"].round(2)
display_vendor_summary["Percent of Total Spending"] = display_vendor_summary["Percent of Total Spending"].round(2)
st.dataframe(display_vendor_summary, use_container_width=True, hide_index=True)

st.subheader("Category Summary")
display_category_summary = category_summary_df.copy()
display_category_summary["Spending Value"] = display_category_summary["Spending Value"].round(2)
display_category_summary["Percent of Total Spending"] = display_category_summary["Percent of Total Spending"].round(2)
st.dataframe(display_category_summary, use_container_width=True, hide_index=True)

st.subheader("Result B: Weekly Spending Plot")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(weekly_df["Week Start"], weekly_df["Spending Value"])
ax.set_xlabel("Week Start")
ax.set_ylabel("Spending")
ax.set_title("Weekly Spending")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# =========================
# DOWNLOAD
# =========================
st.subheader("Download Organised Spending")
download_df = categorized_df[
    ["Transaction Date", "Amount", "Spending Value", "Vendor Name", "Category", "Code", "Details"]
].copy()

st.download_button(
    label="Download organised spending CSV",
    data=make_download_csv(download_df),
    file_name="organized_spending.csv",
    mime="text/csv",
)

with st.expander("Preview organised transactions"):
    preview_df = download_df.copy()
    preview_df["Spending Value"] = preview_df["Spending Value"].round(2)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
