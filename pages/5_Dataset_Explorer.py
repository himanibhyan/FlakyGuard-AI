"""
Page 5: Dataset Explorer for FlakyGuard AI.
Allows interactive filtering, multi-facet searching, and metadata inspection
of the processed TSE22 dataset records without exposing internal leakage fields.
"""

import os
import streamlit as st
import pandas as pd
from src.ui_theme import apply_theme

st.set_page_config(page_title="Dataset Explorer | FlakyGuard AI", page_icon="🗃️", layout="wide")
apply_theme()

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, "full_cleaned_dataset.csv")

@st.cache_data
def load_dataset():
    df = pd.read_csv(CLEANED_DATA_PATH)
    return df

df = load_dataset()

st.title("🗃️ TSE22 Dataset Explorer")
st.caption("Interactive browser for 1,513 deduplicated GitHub issues from Barbosa et al., IEEE TSE 2023.")

st.markdown("---")

# Filter controls
f1, f2, f3, f4 = st.columns([1.5, 1.5, 2, 3])

with f1:
    languages = ["All"] + sorted([l for l in df["language"].dropna().unique()])
    selected_lang = st.selectbox("Language:", languages)

with f2:
    status_options = ["All", "T (Flaky)", "F (False Positive)", "ND (Not Determined)"]
    selected_status = st.selectbox("Flakiness Label:", status_options)

with f3:
    all_causes = sorted([c for c in df["root_cause_category"].dropna().unique() if c not in ["--", "-", "nan"]])
    rc_options = ["All"] + all_causes
    selected_rc = st.selectbox("Root-Cause Category:", rc_options)

with f4:
    search_query = st.text_input("Search in Title / Description:", placeholder="e.g. timeout, race, dead lock, assertion")

# Apply filters
filtered_df = df.copy()

if selected_lang != "All":
    filtered_df = filtered_df[filtered_df["language"] == selected_lang]

if selected_status != "All":
    raw_status = selected_status.split()[0]
    filtered_df = filtered_df[filtered_df["flakiness_status"] == raw_status]

if selected_rc != "All":
    filtered_df = filtered_df[filtered_df["root_cause_category"] == selected_rc]

if search_query.strip():
    q = search_query.strip().lower()
    mask = (
        filtered_df["issue_title"].fillna("").str.lower().str.contains(q) |
        filtered_df["issue_body"].fillna("").str.lower().str.contains(q) |
        filtered_df["problem_description_excerpt"].fillna("").str.lower().str.contains(q) |
        filtered_df["repository"].fillna("").str.lower().str.contains(q)
    )
    filtered_df = filtered_df[mask]

st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records**")

# Display columns
display_cols = [
    "issue_id",
    "repository",
    "language",
    "flakiness_status",
    "root_cause_category",
    "issue_title",
    "year",
    "url_issue"
]

available_cols = [c for c in display_cols if c in filtered_df.columns]

# Rename for clean presentation
rename_display = {
    "issue_id": "ID",
    "repository": "Repository",
    "language": "Language",
    "flakiness_status": "Status",
    "root_cause_category": "Root Cause",
    "issue_title": "Issue Title",
    "year": "Year",
    "url_issue": "GitHub URL"
}

table_view = filtered_df[available_cols].rename(columns=rename_display)

st.dataframe(
    table_view,
    column_config={
        "GitHub URL": st.column_config.LinkColumn("GitHub URL")
    },
    hide_index=True,
    use_container_width=True,
    height=420
)

# Record Details Modal / Expander
st.markdown("---")
st.subheader("🔍 Detailed Record Inspector")

if len(filtered_df) > 0:
    selected_id = st.selectbox("Select Issue ID to View Full Details:", filtered_df["issue_id"].tolist())
    record = filtered_df[filtered_df["issue_id"] == selected_id].iloc[0]

    col_meta1, col_meta2 = st.columns([1, 2])
    with col_meta1:
        st.markdown(f"**Issue ID:** `{record.get('issue_id')}`")
        st.markdown(f"**Repository:** `{record.get('repository')}`")
        st.markdown(f"**Language:** `{record.get('language')}`")
        st.markdown(f"**Flakiness Status:** `{record.get('flakiness_status')}`")
        st.markdown(f"**Root Cause:** `{record.get('root_cause_category')}`")
        st.markdown(f"**Fix Strategy:** `{record.get('fix_strategy_category')}`")
        st.markdown(f"**URL:** [{record.get('url_issue')}]({record.get('url_issue')})")

    with col_meta2:
        st.markdown(f"**Title:** {record.get('issue_title', 'N/A')}")
        st.markdown("**Problem Description Excerpt (Annotator):**")
        st.info(record.get("problem_description_excerpt", "N/A"))
        if pd.notna(record.get("issue_body")) and str(record.get("issue_body")).strip():
            with st.expander("View Full Initial Issue Report Body"):
                st.text(record.get("issue_body"))
else:
    st.info("No records match the current filter criteria.")
