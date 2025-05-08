import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from geopy.distance import geodesic
from streamlit_folium import st_folium
from scipy.spatial.distance import cdist

# Streamlit config
st.set_page_config(
    page_title='REPOSE',
    layout="wide",
    initial_sidebar_state="auto"
)
st.markdown("##### Repose Route Optimization Dashboard")

# Read and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("location_data_roue1.xlsx")
    df = df[df['country'] == 'India']
    df = df[df['state'] == df['state1']]
    df = df.drop_duplicates(subset=["latitude", "longitude"])
    df['route'] = df['name of bo'].str.cat(df['route'], sep='_')
    return df

dff = load_data()

# Filters
from streamlit_dynamic_filters import DynamicFilters
dynamic_filters = DynamicFilters(dff, filters=["state1", "name of bo", "route"])    
dynamic_filters.display_filters(location='sidebar')
df = dynamic_filters.filter_df()

# Determine color column
if df['state1'].nunique() >= 2:
    color_column = 'state1'
elif df['name of bo'].nunique() == 1:
    color_column = 'route'
else:
    color_column = 'name of bo'

# Map center
center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()

# Plotly scatter map
fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    hover_name="name of dealer",
    hover_data=["state", "state1", "name of bo", "route"],
    color=color_column,
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

# Plotly map info
st.markdown("##### Geo Location Overview")
st.markdown("""
##### ℹ️ Notes:
1. **If multiple states are selected**, color is based on **state**.
2. **If single state is selected**, color is based on **BO**.
3. **If single BO is selected**, color is based on **route**.
""")

st.plotly_chart(fig, use_container_width=True)

# State-wise overview
@st.cache_data
def get_state_summary(df):
    return df.groupby('state1').agg(
        number_of_dealers=('name of dealer', 'nunique'),
        number_of_towns=('town', 'nunique'),
        number_of_distributors=('name of distributor', 'nunique'),
        number_of_bo=('name of bo', 'nunique'),
        number_of_front_line_managers=('name of front line manager', 'nunique')
    ).reset_index()

# BO-wise overview
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

    summary['total_km_range'] = summary.apply(
        lambda row: calculate_total_distance(
            df[(df['name of bo'] == row['name of bo']) & 
               (df['state1'] == row['state1']) & 
               (df['route'] == row['route'])]
        ), axis=1)
    return summary

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("##### State-wise Overview")
    st.dataframe(get_state_summary(df), use_container_width=True)

with col2:
    st.markdown("##### BO-wise Overview")
    st.dataframe(get_bo_summary(df), use_container_width=True)

# Route-wise mapping
if df['name of bo'].nunique() != 1:
    st.warning("Please select a single Business Office (BO) to view route-wise mapping.")
else:
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    @st.cache_data
    def sort_coords(coords):
        n = len(coords)
        visited = [False] * n
        route = [0]
        visited[0] = True
        for _ in range(n - 1):
            last = route[-1]
            distances = cdist([coords[last]], coords)[0]
            distances[visited] = np.inf
            next_point = np.argmin(distances)
            route.append(next_point)
            visited[next_point] = True
        return route

    color_list = [
        "blue", "green", "red", "purple", "orange", "darkred",
        "darkblue", "darkgreen", "cadetblue", "darkpurple", "pink", "lightblue",
        "lightgreen", "black"
    ]
    color_idx = 0

    for route_name, group in df.groupby("route"):
        coords = group[["latitude", "longitude"]].values
        if len(coords) < 1:
            continue

        order = sort_coords(coords)
        sorted_group = group.iloc[order].reset_index(drop=True)
        color = color_list[color_idx % len(color_list)]
        color_idx += 1

        for i, row in sorted_group.iterrows():
            icon_color = "blue"
            if i == 0:
                icon_color = "red"
            elif i == len(sorted_group) - 1:
                icon_color = "green"

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=row["name of dealer"],
                tooltip=f"{row['name of dealer']} ({i+1}, {route_name})",
                icon=folium.Icon(color=icon_color)
            ).add_to(m)

        folium.PolyLine(
            locations=sorted_group[["latitude", "longitude"]].values.tolist(),
            color=color,
            weight=4,
            opacity=0.7,
            tooltip=f"Route: {route_name}"
        ).add_to(m)

    st.markdown("##### Optimized Route Mapping")
    st_folium(m, width=1900, height=900)
