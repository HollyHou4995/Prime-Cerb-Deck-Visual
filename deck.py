!uv pip install seaborn

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------
# Data Loader
# ----------------------------------------
def load_and_process_room_stats(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None)
    column_names = ['Hotel Name'] + [f'col {i}' for i in range(1, 22)] + [
        'Occ Rooms Variance', 'Occ% Variance', 'ADR Variance', 'RevPAR Variance', 'Revenue Variance']
    df.columns = column_names
    df = df[['Hotel Name', 'Revenue Variance']].copy()
    df.columns = ['hotel name', 'revenue variance']
    df['state'] = df['hotel name'].str.extract(r'([A-Z]{2})-\d+$')
    df = df[df['hotel name'].str.contains(r'-\d+$', na=False)].copy()
    return df

# ----------------------------------------
# Bar Plot with Dots
# ----------------------------------------
def plot_revenue_variance_with_dot(df: pd.DataFrame, title: str):
    state_revenue = df.groupby("state", as_index=False)["revenue variance"].sum()
    state_revenue = state_revenue.sort_values("revenue variance", ascending=False)
    state_revenue["Variance Category"] = state_revenue["revenue variance"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    palette = {"Positive": "limegreen", "Negative": "red"}

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=state_revenue, x="revenue variance", y="state", hue="Variance Category", palette=palette, dodge=False, legend=False, ax=ax)
    sns.stripplot(data=df, x="revenue variance", y="state", color="darkblue", size=6, alpha=0.6, jitter=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("YoY Revenue Variance ($)")
    ax.set_ylabel("State")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

# ----------------------------------------
# Bar Plot without Dots
# ----------------------------------------
def plot_revenue_variance_no_dot(df: pd.DataFrame, title: str):
    state_revenue = df.groupby("state", as_index=False)["revenue variance"].sum()
    state_revenue = state_revenue.sort_values("revenue variance", ascending=False)
    state_revenue["Variance Category"] = state_revenue["revenue variance"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    palette = {"Positive": "limegreen", "Negative": "red"}

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=state_revenue, x="revenue variance", y="state", hue="Variance Category", palette=palette, dodge=False, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='${:,.0f}', label_type='edge', fontsize=9, padding=3)
    ax.set_title(title)
    ax.set_xlabel("YoY Revenue Variance ($)")
    ax.set_ylabel("State")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

# ----------------------------------------
# Monthly Heatmap
# ----------------------------------------
def plot_monthly_heatmap(monthly_df, new_month_df, month_name, title):
    def format_currency(val):
        return f"${val/1000:.1f}K" if pd.notnull(val) else ""

    state_revenue = new_month_df.groupby("state", as_index=False)["revenue variance"].sum()
    heatmap_df = monthly_df.merge(state_revenue, how="left", left_on="State", right_on="state")
    heatmap_df.drop(columns="state", inplace=True)
    heatmap_df.rename(columns={"revenue variance": month_name}, inplace=True)

    month_cols = [col for col in heatmap_df.columns if col != "State"]
    try:
        month_cols = sorted(month_cols, key=lambda x: pd.to_datetime(x, format="%B").month)
    except:
        pass
    heatmap_df = heatmap_df[["State"] + month_cols]

    total_values = heatmap_df.iloc[:, 1:].sum(numeric_only=True)
    total_row = pd.DataFrame([["Total"] + total_values.tolist()], columns=heatmap_df.columns)
    heatmap_df = pd.concat([total_row, heatmap_df], ignore_index=True)

    heatmap_numeric = heatmap_df.copy()
    for col in heatmap_numeric.columns[1:]:
        heatmap_numeric[col] = pd.to_numeric(heatmap_numeric[col], errors="coerce")

    heatmap_labels = heatmap_numeric.copy()
    for col in heatmap_labels.columns[1:]:
        heatmap_labels[col] = heatmap_labels[col].apply(format_currency)

    cmap = LinearSegmentedColormap.from_list("rwg", ["#d73027", "white", "#1a9850"])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        heatmap_numeric.set_index("State").iloc[1:],
        cmap=cmap,
        annot=heatmap_labels.set_index("State").iloc[1:],
        fmt="",
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "YoY Revenue Variance ($)"},
        ax=ax
    )

    for j, col in enumerate(heatmap_labels.columns[1:]):
        ax.text(j + 0.5, -0.5, heatmap_labels.iloc[0, j + 1], ha="center", va="center", fontsize=10, fontweight="bold", color="black")

    ax.set_title(title, fontsize=16, pad=35)
    ax.set_yticklabels(list(heatmap_df["State"].iloc[1:]), rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return heatmap_df, fig

# ----------------------------------------
# Streamlit App
# ----------------------------------------
st.set_page_config(layout="wide", page_title="Prime/Cerb Visualization Tool")
st.header("Prime/Cerb Revenue Variance Dashboard")

# Step 1: Upload New Month
new_file = st.file_uploader("Step 1: Upload New Month File (CSV only)", type=["csv"], key="new")
new_df = None
if new_file:
    try:
        new_df = load_and_process_room_stats(new_file)
        st.success("✅ New Month Data Loaded and Processed")
        st.dataframe(new_df.head())

        chart_title = st.text_input("Step 2: Enter Title for Bar Charts", value="YoY Preferred Corporate Revenue Variance")

        st.subheader("Bar Chart with Hotel-Level Dots")
        fig_dot = plot_revenue_variance_with_dot(new_df, title=chart_title + " (with Dots)")
        st.pyplot(fig_dot)

        st.subheader("Bar Chart without Hotel-Level Dots")
        fig_nodot = plot_revenue_variance_no_dot(new_df, title=chart_title + " (no Dots)")
        st.pyplot(fig_nodot)

    except Exception as e:
        st.error(f"❌ Failed to process new file: {e}")

# Step 2: Upload Existing Monthly File
existing_df = None
if new_df is not None:
    existing_file = st.file_uploader("Step 3: Upload Existing Monthly File (CSV only)", type=["csv"], key="existing")
    if existing_file:
        try:
            existing_df = pd.read_csv(existing_file)
            st.success("✅ Existing Monthly Data Loaded")
            st.dataframe(existing_df.head())
        except Exception as e:
            st.error(f"❌ Failed to load existing file: {e}")

# Step 3: Generate Heatmap
if new_df is not None and existing_df is not None:
    st.header("Step 4: Generate Monthly Heatmap")

    month_name = st.text_input("Enter New Month Name (e.g., July)", value="July")
    heatmap_title = st.text_input("Enter Heatmap Title", value="Monthly YoY Preferred Corporate Revenue by State")

    if st.button("Generate Heatmap"):
        heatmap_df, fig_heatmap = plot_monthly_heatmap(existing_df, new_df, month_name, title=heatmap_title)
        st.success("✅ Heatmap Created")
        st.pyplot(fig_heatmap)
        st.dataframe(heatmap_df)

        csv = heatmap_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Updated Monthly File (for next month)",
            data=csv,
            file_name="updated_monthly_file.csv",
            mime="text/csv"
        )
