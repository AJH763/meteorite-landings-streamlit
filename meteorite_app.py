"""
Name:       Abrar Jawad Habib
CS230:      Section 3
Data:       Meteorite Landings on Earth as recorded by NASA
URL:        http://localhost:8501

Description:
    This program is an interactive data-explorer for NASA's "Meteorite Landings" dataset.
    It lets users:
        – Explore meteorite landings by year, mass, fall type, and classification.
        – Visualize trends over time with charts (bar, scatter, pie).
        – See geographic patterns on an interactive map (scatter or heatmap).

    Queries include:
        1) How many meteorites landed in each decade, and how does this differ by fall type?
        2) What are the heaviest meteorites in a given time period?
        3) Where are meteorites located on Earth, and how does this change with filters?

NOTE: Portions of this layout and plotting code were drafted with help from
      an AI assistant (ChatGPT) and then adapted to this dataset by Abrar.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Excel file must be in the SAME folder as this script
DATA_PATH = Path(__file__).parent / "Meteorite_Landings.xlsx"


# ---------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------

@st.cache_data
def load_data(nrows: int | None = None) -> pd.DataFrame:
    """
    Load the meteorite landings data from Excel and clean it.

    Returns a DataFrame with:
        - year (int)
        - mass_g (float)
        - fall (str)
        - recclass (str)
        - latitude, longitude (float)
        - decade (int)
        - size_category (str)
    """
    df = pd.read_excel(DATA_PATH, nrows=nrows)

    # Standardize column names to something simple
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "mass" in lc:
            col_map[c] = "mass_g"
        elif c.lower() == "fall":
            col_map[c] = "fall"
        elif "recclass" in lc or "class" in lc:
            col_map[c] = "recclass"
        elif "year" in lc:
            col_map[c] = "year"
        elif "lat" in lc:
            col_map[c] = "latitude"
        elif "lon" in lc or "long" in lc:
            col_map[c] = "longitude"

    df = df.rename(columns=col_map)

    # --- Clean year ---
    if "year" in df.columns:
        if not np.issubdtype(df["year"].dtype, np.number):
            df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year
    else:
        df["year"] = np.nan

    # --- Clean mass ---
    if "mass_g" in df.columns:
        df["mass_g"] = pd.to_numeric(df["mass_g"], errors="coerce")
    else:
        df["mass_g"] = np.nan

    # --- Clean coordinates ---
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without coordinates for mapping
    df["has_location"] = df["latitude"].notna() & df["longitude"].notna()

    # Add decade column  #[COLUMNS]
    df["decade"] = (df["year"] // 10) * 10

    # Add a simple size category using a list comprehension #[LISTCOMP]
    def mass_to_size(m):
        if pd.isna(m):
            return "Unknown"
        if m < 100:
            return "Small (<100 g)"
        if m < 1000:
            return "Medium (100–1000 g)"
        if m < 10000:
            return "Large (1–10 kg)"
        return "Huge (>10 kg)"

    df["size_category"] = [mass_to_size(m) for m in df["mass_g"]]

    # Replace missing text columns with 'Unknown'
    for col in ["fall", "recclass"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


# ---------------------------------------------------------
# Helper functions (satisfy Python feature requirements)
# ---------------------------------------------------------

def filter_by_year_and_mass(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    min_mass: float = 0.0
) -> pd.DataFrame:
    """
    Filter the data frame by a year range and minimum mass.

    Parameters:
        df: input DataFrame
        year_range: (start_year, end_year)
        min_mass: minimum mass in grams (default 0)

    Returns:
        Filtered DataFrame

    #[FUNC2P] function with 2+ params, one with default
    #[FILTER2] two conditions: year range AND min_mass
    """
    start_year, end_year = year_range
    mask = (
        df["year"].between(start_year, end_year, inclusive="both")
        & (df["mass_g"].fillna(0) >= min_mass)
    )
    return df[mask]


def get_top_heaviest(df: pd.DataFrame, n: int = 10):
    """
    Return the top-n heaviest meteorites and the max/min mass.

    #[FUNCRETURN2] returns two values
    #[SORT] uses sort_values
    #[MAXMIN] finds max/min mass
    """
    df_sorted = df.sort_values("mass_g", ascending=False)
    top = df_sorted.head(n)
    max_mass = df["mass_g"].max()
    min_mass = df["mass_g"].min()
    return top, (max_mass, min_mass)


def build_fall_type_dict(df: pd.DataFrame) -> dict:
    """
    Build a dictionary with counts for each fall type.

    #[DICTMETHOD] uses dict.get and .items()
    #[ITERLOOP] iterates through DataFrame rows
    """
    counts: dict[str, int] = {}
    for fall_value in df["fall"]:
        counts[fall_value] = counts.get(fall_value, 0) + 1
    return counts


def generate_fun_fact(df: pd.DataFrame) -> str:
    """
    Simple fun-fact generator based on the filtered data.
    """
    total = len(df)
    if total == 0:
        return "Your current filters remove all meteorites. Try widening the year range or lowering the mass filter!"

    # Choose one of several fact types at random
    choice = random.choice(["heaviest", "oldest", "recent", "class", "size"])
    df_non_null = df.dropna(subset=["mass_g", "year"])

    if choice == "heaviest" and not df_non_null.empty:
        row = df_non_null.loc[df_non_null["mass_g"].idxmax()]
        return (
            f"The heaviest meteorite in your filters is **{row.get('name', 'Unknown')}** "
            f"with **{row['mass_g']:,.0f} g**, recorded in **{int(row['year'])}**."
        )

    if choice == "oldest" and not df_non_null.empty:
        row = df_non_null.loc[df_non_null["year"].idxmin()]
        return (
            f"The oldest meteorite in your view dates back to **{int(row['year'])}**, "
            f"named **{row.get('name', 'Unknown')}**."
        )

    if choice == "recent" and not df_non_null.empty:
        row = df_non_null.loc[df_non_null["year"].idxmax()]
        return (
            f"The most recent meteorite in your filters is from **{int(row['year'])}** "
            f"and has a mass of **{row['mass_g']:,.0f} g**."
        )

    if choice == "class":
        counts = df["recclass"].value_counts()
        top_class = counts.index[0]
        return (
            f"In your current filters, the most common meteorite class is "
            f"**{top_class}** with **{counts.iloc[0]}** samples."
        )

    # size category fact
    size_counts = df["size_category"].value_counts()
    top_size = size_counts.index[0]
    return (
        f"Most meteorites in your view are **{top_size}**. "
        f"Total meteorites shown: **{total:,}**."
    )


# ---------------------------------------------------------
# Charting helpers
# ---------------------------------------------------------

def plot_decade_counts(df: pd.DataFrame):
    """
    Bar chart: number of meteorites per decade by fall type.

    #[CHART1] Bar chart with labels and legend
    """
    # Pivot for counts #[PIVOTTABLE]
    pivot = pd.pivot_table(
        df,
        index="decade",
        columns="fall",
        values="name" if "name" in df.columns else "id",
        aggfunc="count",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Meteorite Landings per Decade by Fall Type")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Count")
    ax.legend(title="Fall Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def plot_mass_vs_year(df: pd.DataFrame):
    """
    Scatter plot: year vs log(mass).

    #[CHART2] Different chart type (scatter)
    #[LAMBDA] uses lambda in transformation
    """
    df = df.dropna(subset=["year", "mass_g"]).copy()
    df = df[df["mass_g"] > 0]

    df["log_mass"] = df["mass_g"].apply(lambda m: np.log10(m))  # lambda

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["year"], df["log_mass"], alpha=0.5)

    ax.set_title("Meteorite Mass over Time (log10 scale)")
    ax.set_xlabel("Year")
    ax.set_ylabel("log10(Mass in grams)")
    plt.tight_layout()
    st.pyplot(fig)


def plot_class_pie(df: pd.DataFrame):
    """
    Pie chart: distribution of meteorite classes (top classes + 'Other').
    """
    if "recclass" not in df.columns:
        st.info("Meteorite class information is not available in this dataset.")
        return

    counts = df["recclass"].value_counts()
    if counts.empty:
        st.info("No data available for the current filters.")
        return

    top = counts.head(8)
    other_sum = counts.iloc[8:].sum()
    labels = list(top.index)
    sizes = list(top.values)

    if other_sum > 0:
        labels.append("Other")
        sizes.append(other_sum)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140
    )
    ax.set_title("Meteorite Class Distribution")
    st.pyplot(fig)


def show_map(df: pd.DataFrame, map_style: str = "Scatter"):
    """
    Show a PyDeck map with meteorite locations.

    #[MAP]
    """
    df_map = df[df["has_location"]].copy()

    # Take a subset for performance
    if len(df_map) > 5000:
        df_map = df_map.sample(5000, random_state=42)

    midpoint = (np.nanmean(df_map["latitude"]), np.nanmean(df_map["longitude"]))

    st.subheader("Meteorite Landings Map")

    if map_style == "Heatmap":
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df_map,
            get_position="[longitude, latitude]",
            aggregation="SUM",
            get_weight="mass_g",
        )
    else:
        # Default scatter style
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[longitude, latitude]",
            get_radius=20000,
            pickable=True,
            get_fill_color="[255, 140, 0, 150]",
        )

    view_state = pdk.ViewState(
        latitude=midpoint[0],
        longitude=midpoint[1],
        zoom=1,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{name}</b><br/>"
                "Class: {recclass}<br/>"
                "Mass: {mass_g} g<br/>"
                "Year: {year}<br/>"
                "Fall: {fall}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )

    st.pydeck_chart(r)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Meteorite Landings Explorer",
        page_icon="🌘",
        layout="wide"
    )

    # #[ST3] Page design features: sidebar with title & progress “animation”
    st.sidebar.title("🚀 Meteorite Explorer")
    st.sidebar.markdown(
        "Use the controls below to explore NASA's meteorite landings dataset."
    )

    # Animated spinner while loading (simple transition)
    with st.spinner("Loading meteorite data..."):
        df = load_data()

    total_count = len(df)
    unique_classes = df["recclass"].nunique() if "recclass" in df.columns else 0

    # Sidebar widgets
    page = st.sidebar.radio(
        "Choose a view",
        ["Overview", "Time Trends", "Mass Explorer", "Map Explorer"]
    )

    # #[ST1] Dropdown / multiselect
    fall_types = sorted(df["fall"].unique())
    selected_fall = st.sidebar.multiselect(
        "Select fall type(s)",
        options=fall_types,
        default=fall_types,
        help="Filter by whether the meteorite was 'Fell', 'Found', etc."
    )

    # #[ST2] Slider
    min_year = int(df["year"].min()) if df["year"].notna().any() else 1800
    max_year = int(df["year"].max()) if df["year"].notna().any() else 2020

    year_range = st.sidebar.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    min_mass = st.sidebar.number_input(
        "Minimum mass (grams)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="Filter out very small meteorites."
    )

    # Apply filters
    df_filtered = df[df["fall"].isin(selected_fall)]
    df_filtered = filter_by_year_and_mass(df_filtered, year_range, min_mass)  # #[FUNCCALL2]

    # Another FUNCCALL2 usage with full df
    _top_all, (max_mass_all, min_mass_all) = get_top_heaviest(df)  # #[FUNCCALL2]

    # Sidebar “animation”: progress bar shows how many meteorites remain after filters
    if total_count > 0:
        frac = len(df_filtered) / total_count
        st.sidebar.caption("Meteorites matching filters")
        st.sidebar.progress(frac)

    if page == "Overview":
        st.title("Meteorite Landings Explorer")
        st.markdown(
            """
            This interactive app uses NASA's **Meteorite Landings** dataset.
            Use the sidebar to filter by **fall type**, **year range**, and **minimum mass**.
            """
        )

        # --- STAT TILES (improved metric section) ---
        st.subheader("Dataset Summary (after filters)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Meteorites (filtered)", f"{len(df_filtered):,}")
        with col2:
            st.metric("Unique Classes", unique_classes)
        with col3:
            st.metric("Max Mass (all data)", f"{max_mass_all:,.0f} g")
        with col4:
            st.metric("Earliest Year", int(df["year"].min()))

        # --- Result Cards: top 3 heaviest in current filters ---
        if not df_filtered.empty:
            st.markdown("### Featured Meteorites (Top 3 by Mass)")

            top3, _ = get_top_heaviest(df_filtered, n=3)
            card_cols = st.columns(3)

            for col, (_, row) in zip(card_cols, top3.iterrows()):
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            padding: 1rem;
                            border-radius: 0.75rem;
                            background-color: #1f2630;
                            border: 1px solid #3a3f47;
                        ">
                            <h4 style="margin-bottom: 0.5rem;">🪐 {row.get('name', 'Unknown')}</h4>
                            <p style="margin-bottom: 0.25rem;">
                                <b>Mass:</b> {row['mass_g']:,.0f} g
                            </p>
                            <p style="margin-bottom: 0.25rem;">
                                <b>Year:</b> {int(row['year']) if not pd.isna(row['year']) else 'Unknown'}
                            </p>
                            <p style="margin-bottom: 0.25rem;">
                                <b>Fall:</b> {row.get('fall', 'Unknown')}
                            </p>
                            <p style="margin-bottom: 0.25rem;">
                                <b>Class:</b> {row.get('recclass', 'Unknown')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("### Sample of Filtered Data")
        st.dataframe(df_filtered.head(20))

        st.markdown("### Fall Type Counts (all data)")
        fall_dict = build_fall_type_dict(df)   # #[DICTMETHOD] + #[ITERLOOP]
        fall_items = sorted(fall_dict.items(), key=lambda x: x[1], reverse=True)

        for k, v in fall_items:
            st.write(f"**{k}**: {v} meteorites")

        st.markdown("### Fun Fact")
        if st.button("Generate fun fact 🎲"):
            fact = generate_fun_fact(df_filtered)
            st.info(fact)

    elif page == "Time Trends":
        st.title("Time Trends")
        st.markdown(
            "This view shows how meteorite landings vary by **decade** and **fall type**."
        )

        if df_filtered.empty:
            st.warning("No data for the selected filters.")
        else:
            tab1, tab2, tab3 = st.tabs(
                ["Decade counts", "Mass over time", "Class distribution"]
            )
            with tab1:
                plot_decade_counts(df_filtered)
            with tab2:
                plot_mass_vs_year(df_filtered)
            with tab3:
                plot_class_pie(df_filtered)

    elif page == "Mass Explorer":
        st.title("Mass Explorer")
        st.markdown(
            "Explore the heaviest meteorites within your selected filters."
        )

        if df_filtered.empty:
            st.warning("No data for the selected filters.")
        else:
            top_n = st.slider("Number of top meteorites to show", 5, 30, 10)
            top_heavy, (max_mass, min_mass) = get_top_heaviest(df_filtered, n=top_n)

            st.write(
                f"Max mass in filtered data: **{max_mass:,.0f} g**, "
                f"Min mass: **{min_mass:,.0f} g**"
            )

            # “card-style” display for the very heaviest one
            heaviest = top_heavy.iloc[0]
            st.markdown(
                f"""
                <div style="
                    padding: 1rem;
                    margin-bottom: 0.75rem;
                    border-radius: 0.75rem;
                    background: linear-gradient(135deg, #ff7a18, #af002d 80%);
                    color: white;
                ">
                    <h3 style="margin-bottom: 0.25rem;">🏆 Heaviest in view: {heaviest.get('name', 'Unknown')}</h3>
                    <p style="margin-bottom: 0.25rem;">
                        <b>Mass:</b> {heaviest['mass_g']:,.0f} g |
                        <b>Year:</b> {int(heaviest['year']) if not pd.isna(heaviest['year']) else 'Unknown'} |
                        <b>Class:</b> {heaviest.get('recclass', 'Unknown')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(
                top_heavy[[
                    col for col in ["name", "mass_g", "year", "fall", "recclass",
                                    "latitude", "longitude"]
                    if col in top_heavy.columns
                ]]
            )

            # Small playful animation when user explores a lot of meteorites
            if top_n >= 25:
                st.balloons()

    elif page == "Map Explorer":
        st.title("Map Explorer")
        st.markdown(
            "See where meteorites have landed on Earth. "
            "Use the sidebar filters to change what is shown."
        )

        if df_filtered[df_filtered["has_location"]].empty:
            st.warning("No locations available for the selected filters.")
        else:
            map_style = st.radio(
                "Map style",
                ["Scatter", "Heatmap"],
                horizontal=True,
                help="Switch between individual points and density heatmap."
            )
            show_map(df_filtered, map_style=map_style)


if __name__ == "__main__":
    main()
