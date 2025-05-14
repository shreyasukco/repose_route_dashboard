import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
from scipy.spatial.distance import cdist
from folium.plugins import AntPath
from numba import njit
from streamlit_dynamic_filters import DynamicFilters
import itertools
import random
import plotly.express as px
import plotly.colors as pc

# Streamlit configuration
st.set_page_config(
    page_title='REPOSE',
    layout="wide",
    initial_sidebar_state="auto"
)
st.markdown("##### Repose Route Optimization Dashboard")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("location_data_roue1.xlsx")
    df = df[df['country'] == 'India']
    df = df[df['state'] == df['state1']]
    df = df.drop_duplicates(subset=["latitude", "longitude"])
    df['route'] = df['name of bo'].astype(str) + '_' + df['route'].astype(str)
    return df

dff = load_data()

# Apply dynamic filters
dynamic_filters = DynamicFilters(dff, filters=["state1", "name of bo", "route"])
dynamic_filters.display_filters(location='sidebar')
df = dynamic_filters.filter_df()
summary_metrics = {
    "States": df['state1'].nunique(),
    "Dealers": df['name of dealer'].nunique(),
    "Towns": df['town'].nunique(),
    "Distributors": df['name of distributor'].nunique(),
    "FLMs": df['name of front line manager'].nunique(),
    "name of bo": df['name of bo'].nunique(),
    "route":df["route"].nunique()
}
# Determine color column for plotting
if df['state1'].nunique() >= 2:
    color_column = 'state1'
elif df['name of bo'].nunique() == 1:
    color_column = 'route'
else:
    color_column = 'name of bo'

# Generate enough unique colors (up to 100 categories)
custom_color_sequence = pc.qualitative.Alphabet + pc.qualitative.Dark24 + pc.qualitative.Light24

# Map center coordinates
center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()

# Plotly scatter map
fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    hover_name="name of dealer",
    hover_data=["state", "state1", "name of bo", "route"],
    color=color_column,
    color_discrete_sequence=custom_color_sequence,
    zoom=10,
    center={"lat": center_lat, "lon": center_lon},
    height=600
)

fig.update_traces(marker=dict(size=12, opacity=0.8))
fig.update_layout(
    mapbox_style="open-street-map",
    margin=dict(r=0, t=0, l=0, b=0),
    paper_bgcolor="#D3D3D3",
    plot_bgcolor="#D3D3D3"
)
st.markdown("""
    <style>
    .metric-box {
        background-color: #003d8f;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        height: 100px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 2px solid white; /* Red border */

    }
    .metric-label {
        font-size: 16px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
# Display map and notes
col1, col2, col3, col4, col5, col6,col7 = st.columns(7, gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">States Covered</div>
            <div class="metric-value">{summary_metrics['States']}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Active Dealers</div>
            <div class="metric-value">{summary_metrics['Dealers']}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Active BOs</div>
            <div class="metric-value">{summary_metrics['name of bo']}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Distributors</div>
            <div class="metric-value">{summary_metrics['Distributors']}</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Front Line Managers </div>
            <div class="metric-value">{summary_metrics['FLMs']}</div>
        </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Towns </div>
            <div class="metric-value">{summary_metrics['Towns']}</div>
        </div>
    """, unsafe_allow_html=True)
with col7:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Routes Covered</div>
            <div class="metric-value">{summary_metrics['route']}</div>
        </div>
    """, unsafe_allow_html=True)
st.write("")
st.markdown("##### Geo Location Overview")
st.markdown("""
##### ℹ️ Notes:
1. **If multiple states are selected**, color is based on **state**.   2. **If single state is selected**, color is based on **BO**.   3. **If single BO is selected**, color is based on **route**.
""")
st.plotly_chart(fig, use_container_width=True)

# Numba-optimized route distance calculation
@njit(fastmath=True)
def route_distance_numba(route, dist_matrix):
    total = 0.0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]
    return total

# State-wise summary
@st.cache_data
def get_state_summary(df):
    return df.groupby('state1').agg(
        number_of_dealers=('name of dealer', 'nunique'),
        number_of_towns=('town', 'nunique'),
        number_of_distributors=('name of distributor', 'nunique'),
        number_of_bo=('name of bo', 'nunique'),
        number_of_front_line_managers=('name of front line manager', 'nunique')
    ).reset_index()

# BO-wise summary
@st.cache_data
def get_bo_summary(df):
    def calculate_total_distance(group_df):
        total = 0
        for i in range(1, len(group_df)):
            point1 = (group_df.iloc[i-1]['latitude'], group_df.iloc[i-1]['longitude'])
            point2 = (group_df.iloc[i]['latitude'], group_df.iloc[i]['longitude'])
            total += geodesic(point1, point2).km
        return round(total + 10)

    summary = df.groupby(['name of bo', 'state1', 'route']).agg(
        number_of_dealers=('name of dealer', 'nunique')
    ).reset_index()

    summary['MIN_km_range'] = summary.apply(
        lambda row: calculate_total_distance(
            df[(df['name of bo'] == row['name of bo']) & 
               (df['state1'] == row['state1']) & 
               (df['route'] == row['route'])]
        ), axis=1)
    return summary

# Display summaries
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("##### BO-wise Overview")
    st.dataframe(get_bo_summary(df), use_container_width=True)
with col2:
    st.markdown("##### State-wise Overview")
    st.dataframe(get_state_summary(df), use_container_width=True)

# Numba-optimized 2-opt algorithm
@njit
def numba_two_opt(route, dist_matrix):
    best = route.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 2, min(len(route), i + 15)):
                a, b, c, d = best[i - 1], best[i], best[j - 1], best[j % len(best)]
                current = dist_matrix[a, b] + dist_matrix[c, d]
                potential = dist_matrix[a, c] + dist_matrix[b, d]
                if potential < current:
                    best[i:j] = best[i:j][::-1]
                    improved = True
    return best

# Route optimization function
@st.cache_data(show_spinner="Optimizing route...")
def optimize_route(coords):
    n = len(coords)
    if n < 2:
        return list(range(n))

    dist_matrix = cdist(coords, coords, metric='euclidean')
    population_size = min(200, max(50, n * 2))
    generations = min(1000, max(100, n * 5))
    mutation_rate = max(0.01, min(0.1, 0.5 / n))

    def create_individual():
        individual = np.random.permutation(n)
        return numba_two_opt(individual, dist_matrix)

    population = [create_individual() for _ in range(population_size)]
    progress_bar = st.progress(0)

    for gen in range(generations):
        population = sorted(population, key=lambda x: route_distance_numba(x, dist_matrix))
        next_gen = population[:10]

        while len(next_gen) < population_size:
            p1, p2 = random.choices(population[:50], k=2)
            a, b = sorted(random.sample(range(n), 2))
            child = np.concatenate([
                p2[~np.isin(p2, p1[a:b])],
                p1[a:b]
            ])
            if random.random() < mutation_rate:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            next_gen.append(numba_two_opt(child, dist_matrix))

        population = next_gen
        progress_bar.progress((gen + 1) / generations)

    progress_bar.empty()
    return min(population, key=lambda x: route_distance_numba(x, dist_matrix))

# Create base map
def create_base_map(center, zoom=12):
    return folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png',
        attr='OpenStreetMap.DE',
        control_scale=True
    )

# Optimized Route Planning
st.markdown("---")
st.markdown("##### Optimized Route Planning")

if df['name of bo'].nunique() != 1:
    st.warning("Please select only one single BO to run optimization.")
else:
    with st.spinner("Generating optimized route..."):
        m = create_base_map([df['latitude'].mean(), df['longitude'].mean()])
        color_cycler = itertools.cycle(px.colors.qualitative.Plotly)

        for route_name, group in df.groupby("route"):
            coords = group[["latitude", "longitude"]].values
            if len(coords) < 1:
                continue

            route_indices = optimize_route(coords)
            sorted_group = group.iloc[route_indices].reset_index(drop=True)

            AntPath(
                locations=sorted_group[["latitude", "longitude"]].values.tolist(),
                color=next(color_cycler),
                weight=6,
                opacity=0.8,
                tooltip=f"Route: {route_name}",
                dash_array=[15, 25]
            ).add_to(m)

            for i, row in sorted_group.iterrows():
                icon_color = "red" if i == 0 else "green" if i == len(sorted_group)-1 else "blue"
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=f"{row['name of dealer']} ({i+1})",
                    icon=folium.Icon(color=icon_color)
                ).add_to(m)

        st_folium(m, width=1900, height=800, use_container_width=True)
