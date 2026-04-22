import io
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Spending Analyzer", layout="wide")

DEFAULT_CATEGORY = "Uncategorised"
LOW_VALUE_CATEGORY = "Low Value Vendors"
REQUIRED_COLUMNS = ["Transaction Date", "Amount", "Code", "Details"]
DEFAULT_MANUAL_CATEGORIES = """Groceries
Bars, Restaurants and Take Outs
Gas Station
Dairys
Café
Tech
Mechanics
Clothing and Grooming
Dentistry and Healthcare
Movies, Audible, Gaming
Public Transport, Uber, Lime
Dance
Direct Bank Transfer
Bottle Shops
UC Printing
Government
Paypal
Hardware Store
The Warehouse and Kmart
Circumstantial One Off
Miscellaneous
Unknown Vendor"""

st.title("Spending Analyzer")
st.caption("Upload transactions, apply known mappings, triage unresolved vendors, and inspect the results.")


# =========================
# HELPERS
# =========================
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


@st.cache_data(show_spinner=False)
def read_single_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if suffix == ".csv":
        df = pd.read_csv(buffer)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(buffer)
    else:
        raise ValueError(f"Unsupported file type for {filename}. Please upload CSV, XLSX, or XLS.")

    df = normalise_columns(df)
    df["Source File"] = filename
    return df


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
    spending_df.loc[spending_df["Vendor Name"].eq(""), "Vendor Name"] = "Unknown Vendor"

    spending_df["Date"] = spending_df["Transaction Date"].dt.normalize()
    spending_df["Week Start"] = spending_df["Transaction Date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    return spending_df


def vendor_totals(spending_df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        spending_df.groupby("Vendor Name", dropna=False)
        .agg(
            Amount=("Spending Value", "sum"),
            Transactions=("Vendor Name", "size"),
        )
        .reset_index()
        .sort_values(["Amount", "Transactions", "Vendor Name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return totals


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def parse_mapping_text(mapping_text: str) -> tuple[dict[str, str], list[str], list[str]]:
    mapping = {}
    errors = []
    warnings = []

    lines = [line for line in mapping_text.splitlines() if line.strip()]
    if not lines:
        return mapping, errors, warnings

    for line_number, raw_line in enumerate(lines, start=1):
        parts = [part.strip() for part in raw_line.split("\t")]

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


def parse_manual_categories(text: str, imported_map: dict[str, str]) -> list[str]:
    user_categories = [line.strip() for line in text.splitlines() if line.strip()]
    imported_categories = sorted(set(imported_map.values()))

    options = []
    for item in [DEFAULT_CATEGORY, LOW_VALUE_CATEGORY] + imported_categories + user_categories:
        if item and item not in options:
            options.append(item)
    return options


def build_assignment_frame(
    totals_df: pd.DataFrame,
    imported_map: dict[str, str],
    manual_map: dict[str, str],
    low_value_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped_mask = totals_df["Vendor Name"].isin(imported_map.keys())
    low_value_mask = totals_df["Amount"] <= low_value_cutoff

    mapped_df = totals_df[mapped_mask].copy()
    low_value_df = totals_df[~mapped_mask & low_value_mask].copy()
    manual_df = totals_df[~mapped_mask & ~low_value_mask].copy()

    for df_name in [mapped_df, low_value_df, manual_df]:
        if not df_name.empty:
            df_name["Share of Total"] = 0.0

    total_spend = totals_df["Amount"].sum()
    if total_spend > 0:
        if not mapped_df.empty:
            mapped_df["Share of Total"] = mapped_df["Amount"] / total_spend * 100
        if not low_value_df.empty:
            low_value_df["Share of Total"] = low_value_df["Amount"] / total_spend * 100
        if not manual_df.empty:
            manual_df["Share of Total"] = manual_df["Amount"] / total_spend * 100

    if not mapped_df.empty:
        mapped_df["Category"] = mapped_df["Vendor Name"].map(imported_map)
    if not low_value_df.empty:
        low_value_df["Category"] = LOW_VALUE_CATEGORY
    if not manual_df.empty:
        manual_df["Category"] = manual_df["Vendor Name"].map(manual_map).fillna(DEFAULT_CATEGORY)

    return manual_df, mapped_df, low_value_df


def apply_final_categories(
    spending_df: pd.DataFrame,
    imported_map: dict[str, str],
    manual_map: dict[str, str],
    low_value_cutoff: float,
) -> pd.DataFrame:
    totals_df = vendor_totals(spending_df)

    final_map = {}

    for _, row in totals_df.iterrows():
        vendor = row["Vendor Name"]
        amount = row["Amount"]

        if vendor in imported_map:
            final_map[vendor] = imported_map[vendor]
        elif amount <= low_value_cutoff:
            final_map[vendor] = LOW_VALUE_CATEGORY
        else:
            final_map[vendor] = manual_map.get(vendor, DEFAULT_CATEGORY)

    categorized_df = spending_df.copy()
    categorized_df["Category"] = categorized_df["Vendor Name"].map(final_map).fillna(DEFAULT_CATEGORY)
    return categorized_df


def vendor_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Category", "Vendor Name"], dropna=False)["Spending Value"]
        .sum()
        .reset_index()
    )

    category_totals = (
        summary.groupby("Category", dropna=False)["Spending Value"]
        .sum()
        .reset_index()
        .rename(columns={"Spending Value": "Category Total"})
    )

    summary = summary.merge(category_totals, on="Category", how="left")
    summary["Percent of Category Spending"] = summary["Spending Value"] / summary["Category Total"] * 100

    summary = summary.sort_values(
        ["Category Total", "Category", "Spending Value", "Vendor Name"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    summary = summary.rename(
        columns={
            "Category": "Category",
            "Vendor Name": "Vendor",
            "Spending Value": "Amount Spent",
        }
    )

    return summary[["Category", "Vendor", "Amount Spent", "Percent of Category Spending"]]


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


def make_weekly_plot(weekly_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_df["Week Start"], weekly_df["Spending Value"], linewidth=2.2)

    ax.set_title("Weekly Spending", fontsize=14, pad=12)
    ax.set_xlabel("Week Start")
    ax.set_ylabel("Spending ($)")
    ax.grid(True, alpha=0.25)

    if not weekly_df.empty:
        date_min = weekly_df["Week Start"].min()
        date_max = weekly_df["Week Start"].max()
        total_days = max((date_max - date_min).days, 1)
        total_weeks = max(math.ceil(total_days / 7), 1)

        if total_weeks <= 16:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        elif total_weeks <= 52:
            interval = math.ceil(total_weeks / 12)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        elif total_weeks <= 156:
            interval = max(1, math.ceil(total_weeks / 10))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))
        else:
            total_months = max(math.ceil(total_days / 30), 1)
            interval = max(1, math.ceil(total_months / 12))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def figure_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# SESSION STATE
# =========================
if "manual_vendor_map_v5" not in st.session_state:
    st.session_state.manual_vendor_map_v5 = {}
if "imported_vendor_map_v5" not in st.session_state:
    st.session_state.imported_vendor_map_v5 = {}
if "uploaded_file_names_v5" not in st.session_state:
    st.session_state.uploaded_file_names_v5 = []


# =========================
# UPLOAD SECTION
# =========================
st.header("Upload")

uploaded_files = st.file_uploader(
    "Upload one or more CSV / Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more transaction files to begin.")
    st.stop()

uploaded_names = sorted([file.name for file in uploaded_files])
if st.session_state.uploaded_file_names_v5 != uploaded_names:
    st.session_state.manual_vendor_map_v5 = {}
    st.session_state.uploaded_file_names_v5 = uploaded_names

raw_frames = []
try:
    for uploaded_file in uploaded_files:
        raw_frames.append(read_single_file(uploaded_file.getvalue(), uploaded_file.name))
    raw_df = pd.concat(raw_frames, ignore_index=True)
    spending_df = prepare_spending_data(raw_df)
except Exception as exc:
    st.error(f"Error reading file: {exc}")
    st.stop()

totals_df = vendor_totals(spending_df)
total_spend = spending_df["Spending Value"].sum()

u1, u2, u3, u4 = st.columns(4)
u1.metric("Number of Transactions", f"{len(raw_df):,}")
u2.metric("Number of Withdrawals", f"{len(spending_df):,}")
u3.metric("Unique Vendors", f"{spending_df['Vendor Name'].nunique():,}")
u4.metric("Total Spending", format_currency(total_spend))

with st.expander("Uploaded files", expanded=False):
    st.write("\n".join(f"- {name}" for name in uploaded_names))


# =========================
# CATEGORISATION SECTION
# =========================
st.header("Categorisation")

st.subheader("Import Vendor Mappings")
st.caption("Copy two columns directly from Excel: vendor in the first column, category in the second. Paste them below.")

with st.form("mapping_import_form_v5"):
    mapping_text = st.text_area(
        "Import Vendor Mappings",
        height=180,
        placeholder="List of Unique Vendors\tList of Unique Vendor's Category\nBp Connect B\tGas Station\nAudible Limi\tMovies, Audible, Gaming",
        label_visibility="collapsed",
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

        st.session_state.imported_vendor_map_v5.update(matched_map)

        for warning in parse_warnings:
            st.warning(warning)

        st.success(f"Applied {len(matched_map):,} pasted mappings.")
        if unmatched_pasted:
            st.warning(f"{len(unmatched_pasted):,} pasted vendors did not match the current dataset exactly and were ignored.")

manual_categories_default = DEFAULT_MANUAL_CATEGORIES
if st.session_state.imported_vendor_map_v5:
    imported_categories_sorted = sorted(set(st.session_state.imported_vendor_map_v5.values()))
    manual_categories_default = "\n".join(
        imported_categories_sorted
        + [c for c in DEFAULT_MANUAL_CATEGORIES.splitlines() if c not in imported_categories_sorted]
    )

st.subheader("Categories")
manual_categories_text = st.text_area(
    "Manual Categories",
    value=manual_categories_default,
    height=140,
    label_visibility="collapsed",
)

st.subheader("Low Value Vendors Cut Off Amount")
low_value_cutoff = st.number_input(
    "Low Value Vendors Cut Off Amount",
    min_value=0.0,
    value=4.0,
    step=1.0,
    label_visibility="collapsed",
)

category_options = parse_manual_categories(manual_categories_text, st.session_state.imported_vendor_map_v5)

manual_df, mapped_df, low_value_df = build_assignment_frame(
    totals_df=totals_df,
    imported_map=st.session_state.imported_vendor_map_v5,
    manual_map=st.session_state.manual_vendor_map_v5,
    low_value_cutoff=low_value_cutoff,
)

current_manual_vendors = set(manual_df["Vendor Name"].tolist())
st.session_state.manual_vendor_map_v5 = {
    k: v for k, v in st.session_state.manual_vendor_map_v5.items()
    if k in current_manual_vendors
}

categorized_df = apply_final_categories(
    spending_df=spending_df,
    imported_map=st.session_state.imported_vendor_map_v5,
    manual_map=st.session_state.manual_vendor_map_v5,
    low_value_cutoff=low_value_cutoff,
)

uncategorised_spend = categorized_df.loc[categorized_df["Category"] == DEFAULT_CATEGORY, "Spending Value"].sum()
auto_categorised_spend = categorized_df.loc[
    categorized_df["Vendor Name"].isin(st.session_state.imported_vendor_map_v5.keys()), "Spending Value"
].sum()
low_value_spend = low_value_df["Amount"].sum()
manually_categorised_spend = categorized_df.loc[
    (~categorized_df["Vendor Name"].isin(st.session_state.imported_vendor_map_v5.keys()))
    & (categorized_df["Category"] != DEFAULT_CATEGORY)
    & (categorized_df["Category"] != LOW_VALUE_CATEGORY),
    "Spending Value",
].sum()
accounted_for_pct = 0.0 if total_spend <= 0 else (total_spend - uncategorised_spend) / total_spend * 100
vendors_uncategorised = categorized_df.loc[categorized_df["Category"] == DEFAULT_CATEGORY, "Vendor Name"].nunique()

st.subheader("Categorisation Stats")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Spending Accounted for", f"{accounted_for_pct:.1f}%")
c2.metric("Spending Auto Categorised", format_currency(auto_categorised_spend))
c3.metric("Spending on Low Stakes Vendors", format_currency(low_value_spend))
c4.metric("Spending Manually Categorised", format_currency(manually_categorised_spend))
c5.metric("Spending Uncategorised", format_currency(uncategorised_spend))
c6.metric("Vendors Uncategorised", f"{vendors_uncategorised:,}")

with st.expander("Manual vendor assignment queue", expanded=True):
    if manual_df.empty:
        st.success("No manual vendors remain above the cutoff.")
    else:
        header_cols = st.columns([1.6, 1.8, 5.5, 2.3])
        header_cols[0].markdown("**Amount ($)**")
        header_cols[1].markdown("**Weight (%)**")
        header_cols[2].markdown("**Vendor Name**")
        header_cols[3].markdown("**Category**")

        for _, row in manual_df.iterrows():
            vendor = row["Vendor Name"]
            amount = row["Amount"]
            share = row["Share of Total"]
            current_value = st.session_state.manual_vendor_map_v5.get(vendor, DEFAULT_CATEGORY)

            cols = st.columns([1.6, 1.8, 5.5, 2.3])
            cols[0].markdown(format_currency(amount))
            cols[1].markdown(f"{share:.2f}%")
            cols[2].markdown(vendor)
            with cols[3]:
                selected = st.selectbox(
                    f"Category for {vendor}",
                    options=category_options,
                    index=category_options.index(current_value) if current_value in category_options else 0,
                    key=f"manual_vendor_assignment_v5_{vendor}",
                    label_visibility="collapsed",
                )
                st.session_state.manual_vendor_map_v5[vendor] = selected

with st.expander("Vendors auto-matched from imported mappings", expanded=False):
    if mapped_df.empty:
        st.info("No imported vendor mappings matched this dataset.")
    else:
        header_cols = st.columns([1.6, 1.8, 5.5, 2.3])
        header_cols[0].markdown("**Amount ($)**")
        header_cols[1].markdown("**Weight (%)**")
        header_cols[2].markdown("**Vendor Name**")
        header_cols[3].markdown("**Category**")

        for _, row in mapped_df.iterrows():
            vendor = row["Vendor Name"]
            amount = row["Amount"]
            share = row["Share of Total"]
            current_value = st.session_state.imported_vendor_map_v5.get(vendor, DEFAULT_CATEGORY)

            cols = st.columns([1.6, 1.8, 5.5, 2.3])
            cols[0].markdown(format_currency(amount))
            cols[1].markdown(f"{share:.2f}%")
            cols[2].markdown(vendor)
            with cols[3]:
                selected = st.selectbox(
                    f"Imported category for {vendor}",
                    options=category_options,
                    index=category_options.index(current_value) if current_value in category_options else 0,
                    key=f"imported_vendor_assignment_v5_{vendor}",
                    label_visibility="collapsed",
                )
                st.session_state.imported_vendor_map_v5[vendor] = selected

with st.expander("Uncategorised Low-value Vendors", expanded=False):
    if low_value_df.empty:
        st.info("No low-value vendors fell below the cutoff.")
    else:
        header_cols = st.columns([1.6, 1.8, 5.5, 2.3])
        header_cols[0].markdown("**Amount ($)**")
        header_cols[1].markdown("**Weight (%)**")
        header_cols[2].markdown("**Vendor Name**")
        header_cols[3].markdown("**Category**")

        for _, row in low_value_df.iterrows():
            vendor = row["Vendor Name"]
            amount = row["Amount"]
            share = row["Share of Total"]
            current_value = st.session_state.manual_vendor_map_v5.get(vendor, LOW_VALUE_CATEGORY)

            cols = st.columns([1.6, 1.8, 5.5, 2.3])
            cols[0].markdown(format_currency(amount))
            cols[1].markdown(f"{share:.2f}%")
            cols[2].markdown(vendor)
            with cols[3]:
                selected = st.selectbox(
                    f"Low value category for {vendor}",
                    options=category_options,
                    index=category_options.index(current_value) if current_value in category_options else 0,
                    key=f"low_value_vendor_assignment_v5_{vendor}",
                    label_visibility="collapsed",
                )
                st.session_state.manual_vendor_map_v5[vendor] = selected

final_override_map = dict(st.session_state.manual_vendor_map_v5)
for vendor, category in st.session_state.imported_vendor_map_v5.items():
    final_override_map[vendor] = category
for vendor in low_value_df["Vendor Name"].tolist():
    if vendor not in final_override_map:
        final_override_map[vendor] = LOW_VALUE_CATEGORY

categorized_df = spending_df.copy()
categorized_df["Category"] = categorized_df["Vendor Name"].map(final_override_map).fillna(DEFAULT_CATEGORY)


# =========================
# RESULTS
# =========================
st.header("Results")

summary_df = vendor_summary_table(categorized_df)
weekly_df = weekly_spending_table(categorized_df)
weekly_fig = make_weekly_plot(weekly_df)
graph_png_bytes = figure_to_png_bytes(weekly_fig)

with st.expander("Accountant-Ready Spending Table", expanded=False):
    display_summary = summary_df.copy()
    display_summary["Amount Spent"] = display_summary["Amount Spent"].round(2)
    display_summary["Percent of Category Spending"] = display_summary["Percent of Category Spending"].round(2)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

with st.expander("Weekly Spending Graph", expanded=False):
    st.pyplot(weekly_fig)


# =========================
# DOWNLOADS
# =========================
st.header("Downloads")

spending_results_df = categorized_df[
    ["Transaction Date", "Amount", "Spending Value", "Vendor Name", "Category", "Code", "Details", "Source File"]
].copy()
mapping_export_df = build_mapping_export(categorized_df)

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        label="Download Vendor Mapping",
        data=make_download_csv(mapping_export_df),
        file_name="vendor_mapping_library.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    st.download_button(
        label="Download Spending Results",
        data=make_download_csv(spending_results_df),
        file_name="organized_spending.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d3:
    st.download_button(
        label="Download Spending Over Time Graph",
        data=graph_png_bytes,
        file_name="weekly_spending_graph.png",
        mime="image/png",
        use_container_width=True,
    )

with st.expander("Preview Vendor Mapping Download", expanded=False):
    st.dataframe(mapping_export_df, use_container_width=True, hide_index=True)

with st.expander("Preview Spending Results Download", expanded=False):
    preview_results = spending_results_df.copy()
    preview_results["Spending Value"] = preview_results["Spending Value"].round(2)
    st.dataframe(preview_results, use_container_width=True, hide_index=True)

with st.expander("Preview Spending Over Time Graph Download", expanded=False):
    st.pyplot(weekly_fig)
