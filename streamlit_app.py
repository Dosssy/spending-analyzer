import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Spending Analyzer", layout="wide")

st.title("Spending Analyzer")
st.caption(
    "Upload your bank transaction file, auto-apply known vendor mappings, "
    "triage the biggest uncategorised vendors first, and stop when the long tail isn't worth the effort."
)


# =========================
# CONSTANTS
# =========================
REQUIRED_COLUMNS = ["Transaction Date", "Amount", "Code", "Details"]
DEFAULT_PRESET_CATEGORIES = (
    "Groceries, Bars, Restaurants and Take Outs, Gas Station, Dairys, Cafe, "
    "Tech, Mechanics, Clothing and Grooming, Dentistry and Healthcare, "
    "Movies, Audible, Gaming, Public Transport, Uber, Lime, Dance, "
    "Direct Bank Transfer, Bottle Shops, UC Printing, Government, Paypal, "
    "Hardware Store, The Warehouse and Kmart, Circumstantial One Off, Miscellaneous"
)


# =========================
# HELPERS
# =========================
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if suffix == ".csv":
        df = pd.read_csv(buffer)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(buffer)
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


@st.cache_data(show_spinner=False)
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
    spending_df.loc[spending_df["Vendor Name"].eq(""), "Vendor Name"] = "Unknown Vendor"

    spending_df["Date"] = spending_df["Transaction Date"].dt.normalize()
    spending_df["Month"] = spending_df["Transaction Date"].dt.to_period("M").astype(str)
    spending_df["Week Start"] = spending_df["Transaction Date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    return spending_df


@st.cache_data(show_spinner=False)
def vendor_totals(spending_df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        spending_df.groupby("Vendor Name", dropna=False)
        .agg(
            Spending_Value=("Spending Value", "sum"),
            Transaction_Count=("Vendor Name", "size"),
        )
        .reset_index()
        .sort_values(["Spending_Value", "Transaction_Count", "Vendor Name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return totals


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def parse_mapping_text(mapping_text: str) -> tuple[dict[str, str], list[str], list[str]]:
    mapping: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []

    lines = [line for line in mapping_text.splitlines() if line.strip()]
    if not lines:
        return mapping, errors, warnings

    for line_number, raw_line in enumerate(lines, start=1):
        parts = [part.strip() for part in raw_line.split("\t")]

        # Allow accidental comma-separated fallback.
        if len(parts) == 1 and "," in raw_line:
            parts = [part.strip() for part in raw_line.split(",", 1)]

        if len(parts) < 2:
            errors.append(f"Line {line_number}: could not find two columns.")
            continue

        vendor = parts[0]
        category = parts[1]

        header_vendor = vendor.casefold() in {"list of unique vendors", "vendor name", "vendor", "unique vendor"}
        header_category = category.casefold() in {"list of unique vendor's category", "category", "vendor category"}
        if line_number == 1 and header_vendor and header_category:
            warnings.append("Header row detected and ignored.")
            continue

        if not vendor:
            errors.append(f"Line {line_number}: vendor is blank.")
            continue
        if not category:
            errors.append(f"Line {line_number}: category is blank for vendor '{vendor}'.")
            continue

        mapping[vendor] = category

    return mapping, errors, warnings


def build_category_options(default_category: str, preset_categories_text: str, low_value_category: str) -> list[str]:
    raw = [x.strip() for x in preset_categories_text.split(",") if x.strip()]
    options: list[str] = []
    for item in [default_category] + raw + [low_value_category]:
        if item and item not in options:
            options.append(item)
    return options


def apply_category_mapping(
    spending_df: pd.DataFrame,
    manual_map: dict[str, str],
    imported_map: dict[str, str],
    default_category: str,
    low_value_cutoff: float,
    low_value_category: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    totals_df = vendor_totals(spending_df)

    manual_queue_df = totals_df[totals_df["Spending_Value"] > low_value_cutoff].copy()
    auto_queue_df = totals_df[totals_df["Spending_Value"] <= low_value_cutoff].copy()

    final_map: dict[str, str] = {}

    # Low-value vendors go straight to the tail bucket.
    for _, row in auto_queue_df.iterrows():
        final_map[row["Vendor Name"]] = low_value_category

    # For higher-value vendors, imported rules come first, then manual overrides.
    for _, row in manual_queue_df.iterrows():
        vendor = row["Vendor Name"]
        final_map[vendor] = imported_map.get(vendor, default_category)
        if vendor in manual_map:
            final_map[vendor] = manual_map[vendor]

    categorized_df = spending_df.copy()
    categorized_df["Category"] = categorized_df["Vendor Name"].map(final_map).fillna(default_category)

    unmatched_queue_df = manual_queue_df[~manual_queue_df["Vendor Name"].isin(imported_map.keys())].copy()
    matched_queue_df = manual_queue_df[manual_queue_df["Vendor Name"].isin(imported_map.keys())].copy()

    return categorized_df, matched_queue_df, unmatched_queue_df, auto_queue_df


def vendor_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Category", "Vendor Name"], dropna=False)["Spending Value"]
        .sum()
        .reset_index()
        .sort_values("Spending Value", ascending=False)
        .reset_index(drop=True)
    )
    total = summary["Spending Value"].sum()
    summary["Percent of Total Spending"] = 0.0 if total <= 0 else summary["Spending Value"] / total * 100
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
    summary["Percent of Total Spending"] = 0.0 if total <= 0 else summary["Spending Value"] / total * 100
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


def build_mapping_export(categorized_df: pd.DataFrame) -> pd.DataFrame:
    export_df = (
        categorized_df.groupby("Vendor Name", dropna=False)["Category"]
        .agg(lambda x: x.iloc[0])
        .reset_index()
        .sort_values("Vendor Name")
        .reset_index(drop=True)
    )
    export_df.columns = ["List of Unique Vendors", "List of Unique Vendor's Category"]
    return export_df


# =========================
# SESSION STATE
# =========================
if "manual_vendor_map" not in st.session_state:
    st.session_state.manual_vendor_map = {}
if "imported_vendor_map" not in st.session_state:
    st.session_state.imported_vendor_map = {}
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None


# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload a transaction file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    raw_df = read_uploaded_file(file_bytes, uploaded_file.name)
    spending_df = prepare_spending_data(raw_df)
except Exception as exc:
    st.error(f"Error reading file: {exc}")
    st.stop()

current_filename = uploaded_file.name
if st.session_state.last_uploaded_filename != current_filename:
    st.session_state.manual_vendor_map = {}
    st.session_state.last_uploaded_filename = current_filename

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
left_control, right_control = st.columns(2)

with left_control:
    preset_categories_text = st.text_input(
        "Preset categories (comma separated)",
        value=DEFAULT_PRESET_CATEGORIES,
    )
    default_category = st.text_input(
        "Default category for untouched vendors",
        value="Uncategorised",
    )

with right_control:
    low_value_cutoff = st.number_input(
        "Auto-category cutoff amount",
        min_value=0.0,
        value=4.0,
        step=1.0,
        help="Any vendor with total spending at or below this amount will be automatically categorised.",
    )
    low_value_category = st.text_input(
        "Automatic category for low-value vendors",
        value="Miscellaneous",
    )

category_options = build_category_options(default_category, preset_categories_text, low_value_category)


# =========================
# IMPORT EXISTING MAPPINGS
# =========================
st.subheader("Import Existing Vendor Mappings")
st.caption(
    "Copy two columns directly from Excel: vendor in the first column, category in the second. "
    "Paste them below, then click Apply pasted mappings."
)

with st.form("mapping_import_form"):
    mapping_text = st.text_area(
        "Paste Excel vendor-category mappings",
        height=180,
        placeholder="List of Unique Vendors\tList of Unique Vendor's Category\nBp Connect B\tGas Station\nAudible Limi\tMovies, Audible, Gaming",
    )
    apply_mapping_button = st.form_submit_button("Apply pasted mappings")

if apply_mapping_button:
    parsed_map, parse_errors, parse_warnings = parse_mapping_text(mapping_text)

    if parse_errors:
        st.error("Could not apply pasted mappings:\n- " + "\n- ".join(parse_errors[:10]))
    else:
        current_vendors = set(spending_df["Vendor Name"].unique().tolist())
        matched_map = {k: v for k, v in parsed_map.items() if k in current_vendors}
        unmatched_pasted = [k for k in parsed_map.keys() if k not in current_vendors]

        st.session_state.imported_vendor_map.update(matched_map)

        if parse_warnings:
            for warning in parse_warnings:
                st.warning(warning)

        st.success(f"Applied {len(matched_map):,} pasted mappings.")
        if unmatched_pasted:
            st.warning(
                f"{len(unmatched_pasted):,} pasted vendors did not match the current dataset exactly. "
                f"They were ignored."
            )

if st.session_state.imported_vendor_map:
    with st.expander("Current imported mapping library", expanded=False):
        imported_display = pd.DataFrame(
            {
                "Vendor Name": list(st.session_state.imported_vendor_map.keys()),
                "Category": list(st.session_state.imported_vendor_map.values()),
            }
        ).sort_values("Vendor Name")
        st.dataframe(imported_display, use_container_width=True, hide_index=True)
        if st.button("Clear imported mappings"):
            st.session_state.imported_vendor_map = {}
            st.rerun()


# =========================
# APPLY CATEGORIES
# =========================
categorized_df, matched_queue_df, unmatched_queue_df, auto_queue_df = apply_category_mapping(
    spending_df=spending_df,
    manual_map=st.session_state.manual_vendor_map,
    imported_map=st.session_state.imported_vendor_map,
    default_category=default_category,
    low_value_cutoff=low_value_cutoff,
    low_value_category=low_value_category,
)

# Remove stale manual mappings when current queue changes.
current_manual_vendors = set(unmatched_queue_df["Vendor Name"].tolist())
st.session_state.manual_vendor_map = {
    k: v for k, v in st.session_state.manual_vendor_map.items() if k in current_manual_vendors
}

# Reapply after stale cleanup.
categorized_df, matched_queue_df, unmatched_queue_df, auto_queue_df = apply_category_mapping(
    spending_df=spending_df,
    manual_map=st.session_state.manual_vendor_map,
    imported_map=st.session_state.imported_vendor_map,
    default_category=default_category,
    low_value_cutoff=low_value_cutoff,
    low_value_category=low_value_category,
)

vendor_summary_df = vendor_summary_table(categorized_df)
category_summary_df = category_summary_table(categorized_df)
weekly_df = weekly_spending_table(categorized_df)


total_spend = categorized_df["Spending Value"].sum()
categorized_spend = categorized_df[categorized_df["Category"] != default_category]["Spending Value"].sum()
uncategorized_spend = total_spend - categorized_spend
coverage_pct = 0.0 if total_spend <= 0 else categorized_spend / total_spend * 100


# =========================
# TRIAGE DASHBOARD
# =========================
st.subheader("Categorisation Coverage")
cv1, cv2, cv3, cv4 = st.columns(4)
cv1.metric("Spend Already Categorised", format_currency(categorized_spend))
cv2.metric("Spend Still Uncategorised", format_currency(uncategorized_spend))
cv3.metric("Coverage", f"{coverage_pct:.1f}%")
cv4.metric("Unmatched Vendors Left", f"{len(unmatched_queue_df):,}")

st.caption(
    "This is the real stopping rule. When the remaining uncategorised spend gets small enough, "
    "you can stop with confidence."
)

st.subheader("Vendor Prioritisation Queue")
pq1, pq2, pq3, pq4 = st.columns(4)
pq1.metric("Imported Rules Applied", f"{len(matched_queue_df):,}")
pq2.metric("Manual Vendors Remaining", f"{len(unmatched_queue_df):,}")
pq3.metric("Spend in Manual Queue", format_currency(unmatched_queue_df["Spending_Value"].sum()))
pq4.metric("Spend Auto-Sent to Tail Bucket", format_currency(auto_queue_df["Spending_Value"].sum()))


# =========================
# CATEGORY ASSIGNMENT UI
# =========================
st.subheader("Assign Categories to Highest-Value Unmatched Vendors First")
st.caption(
    "Only unmatched vendors are shown below. They are ordered by total spend so each click resolves the most money possible."
)

with st.expander("Manual vendor assignment queue", expanded=True):
    if len(unmatched_queue_df) == 0:
        st.success("No unmatched vendors remain above the cutoff.")
    else:
        for _, row in unmatched_queue_df.iterrows():
            vendor = row["Vendor Name"]
            spend_amount = row["Spending_Value"]
            transaction_count = int(row["Transaction_Count"])
            current_value = st.session_state.manual_vendor_map.get(vendor, default_category)

            left, middle, right_a, right_b = st.columns([2.2, 5.5, 1.8, 1.2])

            with left:
                selected = st.selectbox(
                    label="Category",
                    options=category_options,
                    index=category_options.index(current_value) if current_value in category_options else 0,
                    key=f"manual_vendor_category_{vendor}",
                    label_visibility="collapsed",
                )
                st.session_state.manual_vendor_map[vendor] = selected

            with middle:
                st.markdown(f"**{vendor}**")

            with right_a:
                st.markdown(f"**{format_currency(spend_amount)}**")

            with right_b:
                st.markdown(f"**{transaction_count} txns**")

if len(matched_queue_df) > 0:
    with st.expander("Vendors auto-matched from imported mappings", expanded=False):
        matched_display = matched_queue_df.copy()
        matched_display["Category"] = matched_display["Vendor Name"].map(st.session_state.imported_vendor_map)
        matched_display["Spending_Value"] = matched_display["Spending_Value"].round(2)
        matched_display.columns = ["Vendor Name", "Spending Value", "Transaction Count", "Assigned Category"]
        st.dataframe(matched_display, use_container_width=True, hide_index=True)

if len(auto_queue_df) > 0:
    with st.expander("Auto-categorised low-value vendors", expanded=False):
        auto_display = auto_queue_df.copy()
        auto_display["Assigned Category"] = low_value_category
        auto_display["Spending_Value"] = auto_display["Spending_Value"].round(2)
        auto_display.columns = ["Vendor Name", "Spending Value", "Transaction Count", "Assigned Category"]
        st.dataframe(auto_display, use_container_width=True, hide_index=True)


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
# DOWNLOADS
# =========================
st.subheader("Downloads")
download_df = categorized_df[
    ["Transaction Date", "Amount", "Spending Value", "Vendor Name", "Category", "Code", "Details"]
].copy()

mapping_export_df = build_mapping_export(categorized_df)

left_dl, right_dl = st.columns(2)
with left_dl:
    st.download_button(
        label="Download organised spending CSV",
        data=make_download_csv(download_df),
        file_name="organized_spending.csv",
        mime="text/csv",
    )

with right_dl:
    st.download_button(
        label="Download updated vendor mapping CSV",
        data=make_download_csv(mapping_export_df),
        file_name="vendor_mapping_library.csv",
        mime="text/csv",
    )

with st.expander("Preview organised transactions"):
    preview_df = download_df.copy()
    preview_df["Spending Value"] = preview_df["Spending Value"].round(2)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

with st.expander("Preview exportable vendor mapping library"):
    st.dataframe(mapping_export_df, use_container_width=True, hide_index=True)
