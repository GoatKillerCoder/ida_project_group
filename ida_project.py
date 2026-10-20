
# -*- coding: utf-8 -*-
"""
IDA Project: Census 2022 Sample Data Analysis
Streamlit web application
"""

from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IDA Project | Census 2022 Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Census 2022 Sample Data Analysis")
st.caption("IDA Project | Data cleaning, exploration and visualisation")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

CSV_PATH = Path(__file__).parent / "Census2022sample_F19.csv"


MAX_SAMPLE_ROWS = 100_000
MAX_CATEGORIES = 20
MAX_CROSSTAB_LEVELS = 30
MAX_HEATMAP_CELLS = 600
MAX_STATS_CELLS = 2_000_000


# cache_resource avoids pickling/copying the large DataFrame on every rerun;
# the cached frames are never modified after they are returned.
@st.cache_resource(show_spinner="Loading dataset...")
def load_data(file_path):
    last_error = None

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


try:
    if not CSV_PATH.exists():
        st.error(
            "Dataset not found. Make sure "
            "'Census2022sample_F19.csv' is in the "
            "same GitHub repository folder as ida_project.py."
        )
        st.stop()

    # Check if Git LFS pointer text was downloaded instead
    with open(CSV_PATH, "rb") as file:
        first_line = file.readline().decode(
            "utf-8", errors="replace"
        ).strip()

    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        st.error(
            "The dataset is a Git LFS pointer, not the actual CSV. "
            "Upload the real CSV file to the repository or use "
            "a supported external data source."
        )
        st.stop()

    df = load_data(str(CSV_PATH))

except pd.errors.EmptyDataError:
    st.error("The CSV file is empty.")
    st.stop()

except pd.errors.ParserError as error:
    st.error(f"Could not parse the CSV file: {error}")
    st.stop()

except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


# --------------------------------------------------
# INITIAL DATA OVERVIEW
# --------------------------------------------------

st.header("1. Initial Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Initial rows", f"{len(df):,}")

with col2:
    st.metric("Initial columns", f"{len(df.columns):,}")

with col3:
    st.metric(
        "Missing values",
        f"{int(df.isna().sum().sum()):,}"
    )

with st.expander("Preview original dataset"):
    st.dataframe(df.head(10), width="stretch")

with st.expander("Original data types and information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Non-null values": df.notna().sum().values,
        "Missing values": df.isna().sum().values
    })
    st.dataframe(info_df, width="stretch")


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

st.header("2. Data Cleaning")

@st.cache_resource(show_spinner="Cleaning data...")
def clean_data(raw):
    """Clean the raw census data. Returns (cleaned_df, duplicates_removed)."""

    data = raw.copy()

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # 1. Convert text columns that are really numeric BEFORE filling gaps,
    #    otherwise they would be filled with "Unknown" instead of the median.
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            continue

        converted = pd.to_numeric(data[col], errors="coerce")
        original_valid = int(data[col].notna().sum())

        if original_valid and converted.notna().sum() / original_valid >= 0.95:
            data[col] = converted

    # 2. Fill missing values (median for numeric, "Unknown" otherwise)
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            median_value = data[col].median()

            if pd.notna(median_value):
                data[col] = data[col].fillna(median_value)
        else:
            data[col] = data[col].fillna("Unknown")

    # 3. Remove duplicate rows
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates()

    # 4. Validate ranges when the columns exist
    if "derh_hhage" in data.columns:
        data = data[data["derh_hhage"].between(0, 100)]

    if "derh_hsize" in data.columns:
        data = data[data["derh_hsize"].between(1, 20)]

    data = data.copy()

    # 5. Convert household head sex codes to labels
    if "derh_hhsex" in data.columns:
        data["derh_hhsex"] = (
            data["derh_hhsex"]
            .replace({1: "Male", 2: "Female"})
            .astype(str)
        )

    return data, duplicates


initial_rows = len(df)

try:
    df, duplicates_removed = clean_data(df)
except Exception as error:
    st.error(f"Data cleaning failed: {error}")
    st.stop()

if df.empty:
    st.error(
        "No rows remain after cleaning. "
        "Check the validation rules against the dataset."
    )
    st.stop()

st.success("Data cleaning completed.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Cleaned rows", f"{len(df):,}")

with col2:
    st.metric(
        "Rows removed",
        f"{initial_rows - len(df):,}"
    )

with col3:
    st.metric("Duplicates removed", f"{duplicates_removed:,}")

with st.expander("Cleaned dataset preview"):
    st.dataframe(df.head(20), width="stretch")

with st.expander("Summary statistics"):
    stats_source = df

    if df.size > MAX_STATS_CELLS:
        stats_source = df.sample(
            n=max(1, MAX_STATS_CELLS // max(1, df.shape[1])),
            random_state=42
        )
        st.caption(
            f"Dataset is large; statistics are based on a random "
            f"sample of {len(stats_source):,} rows."
        )

    st.dataframe(
        stats_source.describe(include="all").T,
        width="stretch"
    )

with st.expander("Remaining missing values"):
    missing_df = df.isna().sum().rename(
        "Missing values"
    ).to_frame()

    st.dataframe(missing_df, width="stretch")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def show_count_chart(
    data,
    column,
    title,
    xlabel,
    ylabel="Count",
    hue=None,
    figsize=(9, 5),
    discrete=False
):
    """Display a Seaborn count chart in Streamlit."""

    if column not in data.columns:
        st.warning(
            f"Cannot display '{title}': "
            f"column '{column}' is missing."
        )
        return

    if data[column].dropna().empty:
        st.warning(
            f"Cannot display '{title}': "
            "the column contains no usable values."
        )
        return

    if hue is not None and hue not in data.columns:
        hue = None

    # Aggregate first so only a small table is plotted, not every row
    keys = [column] if hue is None else [column, hue]
    counts = data.groupby(keys, observed=True).size().rename("count")
    counts = counts.reset_index()

    top_x = (
        counts.groupby(column)["count"].sum()
        .nlargest(MAX_CATEGORIES).index
    )
    counts = counts[counts[column].isin(top_x)]

    if hue is not None:
        top_hue = (
            counts.groupby(hue)["count"].sum()
            .nlargest(MAX_CATEGORIES // 2).index
        )
        counts = counts[counts[hue].isin(top_hue)]

    counts = counts.sort_values(keys)
    counts[column] = counts[column].astype(str)

    if hue is not None:
        counts[hue] = counts[hue].astype(str)

    if data[column].nunique() > MAX_CATEGORIES:
        st.caption(
            f"Showing the {MAX_CATEGORIES} most frequent values of "
            f"'{column}'."
        )

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=counts,
        x=column,
        y="count",
        hue=hue,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_histogram(
    data,
    column,
    title,
    xlabel,
    bins=20
):
    """Display a histogram in Streamlit."""

    if column not in data.columns:
        st.warning(
            f"Cannot display '{title}': "
            f"column '{column}' is missing."
        )
        return

    values = pd.to_numeric(
        data[column], errors="coerce"
    ).dropna()

    if values.empty:
        st.warning(
            f"Cannot display '{title}': "
            "there are no numeric values."
        )
        return

    if len(values) > MAX_SAMPLE_ROWS:
        values = values.sample(MAX_SAMPLE_ROWS, random_state=42)
        st.caption(
            f"Histogram based on a random sample of "
            f"{MAX_SAMPLE_ROWS:,} rows."
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        values,
        bins=bins,
        kde=(len(values) <= 20_000 and values.nunique() > 1),
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_crosstab(data, row, column, title):
    """Display a cross-tabulation and its heatmap."""

    if row not in data.columns or column not in data.columns:
        st.warning(
            f"Cannot display '{title}': "
            "one or more required columns are missing."
        )
        return

    subset = data[[row, column]].dropna()
    truncated = False

    for col in (row, column):
        levels = subset[col].value_counts()

        if len(levels) > MAX_CROSSTAB_LEVELS:
            keep = levels.nlargest(MAX_CROSSTAB_LEVELS).index
            subset = subset[subset[col].isin(keep)]
            truncated = True

    table = pd.crosstab(subset[row], subset[column])

    st.subheader(title)

    if truncated:
        st.caption(
            f"Limited to the {MAX_CROSSTAB_LEVELS} most frequent "
            "values of each variable."
        )

    st.dataframe(table, width="stretch")

    if table.empty:
        return

    if table.size > MAX_HEATMAP_CELLS:
        st.caption("Heatmap skipped because the table is too large.")
        return

    fig, ax = plt.subplots(
        figsize=(
            max(7, min(14, table.shape[1] + 4)),
            max(4, min(9, table.shape[0] * 0.4 + 2))
        )
    )

    sns.heatmap(
        table,
        annot=(table.size <= 100),
        fmt="g",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel(column)
    ax.set_ylabel(row)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.header("3. Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs([
    "Demographics",
    "Hunger and Water",
    "Housing and Tenure"
])


# --------------------------------------------------
# DEMOGRAPHICS
# --------------------------------------------------

with tab1:
    st.subheader("Age Distribution of Household Heads")

    show_histogram(
        df,
        "derh_hhage",
        "Age Distribution of Household Heads",
        "Age of Household Head",
        bins=20
    )

    st.subheader("Household Size Distribution")

    show_histogram(
        df,
        "derh_hsize",
        "Household Size Distribution",
        "Household Size",
        bins=10
    )

    st.subheader("Household Head Sex Distribution")

    show_count_chart(
        df,
        "derh_hhsex",
        "Household Head Sex Distribution",
        "Sex"
    )

    if "derh_hhsex" in df.columns:
        st.write("Household head sex counts")
        st.dataframe(
            df["derh_hhsex"]
            .value_counts(dropna=False)
            .rename_axis("Sex")
            .reset_index(name="Count"),
            width="stretch"
        )


# --------------------------------------------------
# HUNGER AND WATER
# --------------------------------------------------

with tab2:
    st.subheader("Adult Hunger Frequency")

    show_count_chart(
        df,
        "a4_adult_hunger",
        "Adult Hunger Frequency",
        "Adult Hunger Code"
    )

    st.subheader("Child Hunger Frequency")

    show_count_chart(
        df,
        "a5_child_hunger",
        "Child Hunger Frequency",
        "Child Hunger Code"
    )

    st.subheader("Access to Piped Water")

    show_count_chart(
        df,
        "h05_waterpiped",
        "Access to Piped Water",
        "Water Piped Code"
    )

    if "h05_waterpiped" in df.columns:
        st.write("Piped water counts")
        st.dataframe(
            df["h05_waterpiped"]
            .value_counts(dropna=False)
            .rename_axis("Water Piped Code")
            .reset_index(name="Count"),
            width="stretch"
        )

    show_crosstab(
        df,
        "derh_hhsex",
        "h05_waterpiped",
        "Household Head Sex vs Access to Piped Water"
    )

    show_crosstab(
        df,
        "derh_hsize",
        "a4_adult_hunger",
        "Household Size vs Adult Hunger"
    )

    show_crosstab(
        df,
        "derh_hhsex",
        "a4_adult_hunger",
        "Household Head Sex vs Adult Hunger"
    )


# --------------------------------------------------
# HOUSING AND TENURE
# --------------------------------------------------

with tab3:
    show_crosstab(
        df,
        "h02_maindwelling",
        "h03_tenure",
        "Dwelling Type vs Tenure"
    )

    show_count_chart(
        df,
        "h02_maindwelling",
        "Main Dwelling Type Distribution",
        "Dwelling Type Code",
        hue="h03_tenure" if "h03_tenure" in df.columns else None,
        figsize=(12, 6)
    )


# --------------------------------------------------
# SUMMARY ANALYSIS
# --------------------------------------------------

st.header("4. Summary Analysis")

if all(
    col in df.columns
    for col in ["derh_hhsex", "derh_hsize"]
):
    st.subheader("Mean Household Size by Sex of Household Head")

    mean_size = (
        df.groupby("derh_hhsex")["derh_hsize"]
        .mean()
        .rename("Mean Household Size")
        .reset_index()
    )

    st.dataframe(mean_size, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=mean_size,
        x="derh_hhsex",
        y="Mean Household Size",
        ax=ax
    )

    ax.set_title("Mean Household Size by Sex of Household Head")
    ax.set_xlabel("Sex of Household Head")
    ax.set_ylabel("Mean Household Size")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(
        "Mean household size by sex cannot be calculated "
        "because one or more required columns are missing."
    )


if all(
    col in df.columns
    for col in ["h03_tenure", "derh_hhage"]
):
    st.subheader("Average Age of Household Head by Tenure Type")

    mean_age = (
        df.groupby("h03_tenure")["derh_hhage"]
        .mean()
        .rename("Average Age")
        .reset_index()
    )

    st.dataframe(mean_age, width="stretch")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=mean_age,
        x="h03_tenure",
        y="Average Age",
        ax=ax
    )

    ax.set_title("Average Age of Household Head by Tenure Type")
    ax.set_xlabel("Tenure Type")
    ax.set_ylabel("Average Age")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(
        "Average age by tenure cannot be calculated "
        "because one or more required columns are missing."
    )


# --------------------------------------------------
# DOWNLOAD CLEANED DATA
# --------------------------------------------------

st.header("5. Download Cleaned Dataset")

@st.cache_resource(show_spinner="Preparing download...")
def to_csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8")


csv_data = to_csv_bytes(df)

st.download_button(
    label="Download cleaned CSV",
    data=csv_data,
    file_name="Census2022sample_F19_cleaned.csv",
    mime="text/csv"
)

st.caption(
    "Analysis is based on the uploaded Census 2022 sample. "
    "Interpret category codes using the dataset's official codebook."
)
