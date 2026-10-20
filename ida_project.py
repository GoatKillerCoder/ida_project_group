
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


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


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
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Original data types and information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Non-null values": df.notna().sum().values,
        "Missing values": df.isna().sum().values
    })
    st.dataframe(info_df, use_container_width=True)


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

st.header("2. Data Cleaning")

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

initial_rows = len(df)

# Fill missing numeric values using column medians
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    median_value = df[col].median()

    if pd.notna(median_value):
        df[col] = df[col].fillna(median_value)

# Fill missing categorical values
categorical_cols = df.select_dtypes(
    exclude=[np.number]
).columns

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# Remove duplicate rows
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# Validate household head age when available
if "derh_hhage" in df.columns:
    df["derh_hhage"] = pd.to_numeric(
        df["derh_hhage"], errors="coerce"
    )

    df = df[
        df["derh_hhage"].between(0, 100)
    ].copy()

# Validate household size when available
if "derh_hsize" in df.columns:
    df["derh_hsize"] = pd.to_numeric(
        df["derh_hsize"], errors="coerce"
    )

    df = df[
        df["derh_hsize"].between(1, 20)
    ].copy()

# Convert household head sex codes to labels
if "derh_hhsex" in df.columns:
    df["derh_hhsex"] = df["derh_hhsex"].replace({
        1: "Male",
        2: "Female"
    })

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
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Summary statistics"):
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

with st.expander("Remaining missing values"):
    missing_df = df.isna().sum().rename(
        "Missing values"
    ).to_frame()

    st.dataframe(missing_df, use_container_width=True)


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

    fig, ax = plt.subplots(figsize=figsize)

    sns.countplot(
        data=data,
        x=column,
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

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        values,
        bins=bins,
        kde=(len(values) > 1 and values.nunique() > 1),
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

    table = pd.crosstab(data[row], data[column])

    st.subheader(title)
    st.dataframe(table, use_container_width=True)

    if not table.empty:
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
            use_container_width=True
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
            use_container_width=True
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

    st.dataframe(mean_size, use_container_width=True)

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

    st.dataframe(mean_age, use_container_width=True)

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

csv_data = df.to_csv(index=False).encode("utf-8")

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
