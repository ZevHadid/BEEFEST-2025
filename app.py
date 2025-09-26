import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import calendar
import numpy as np

# --- Main App Configuration ---
st.set_page_config(
    page_title="Bangladesh Crime Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load the Data ---
data_file_path = "Dataset 3 - Bangladesh Crime Dataset.csv"
df = pd.read_csv(data_file_path)

# Normalize weekday names
df['incident_weekday'] = df['incident_weekday'].astype(str).str.strip().str.capitalize()
weekday_map = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
    "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"
}
df['incident_weekday'] = df['incident_weekday'].replace(weekday_map)

# --- Title and Introduction ---
st.title("Interactive Crime Data Analysis in Bangladesh")
st.markdown("### A data exploration dashboard built with Streamlit.")
st.write(
    "This application provides an interactive platform to analyze crime data in Bangladesh. "
    "Use the sidebar to filter data and interact with the visualizations."
)

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")

# Division filter
unique_divisions = sorted(df['incident_division'].dropna().unique())
selected_division = st.sidebar.selectbox(
    "Select Division",
    options=["All"] + unique_divisions,
    index=0
)
if selected_division == "All":
    selected_divisions = unique_divisions
else:
    selected_divisions = [selected_division]

# Crime type filter
unique_crimes = sorted(df['crime'].dropna().unique())
selected_crime = st.sidebar.selectbox(
    "Select Crime Type",
    options=["All"] + unique_crimes,
    index=0
)
if selected_crime == "All":
    selected_crimes = unique_crimes
else:
    selected_crimes = [selected_crime]

# Month filter
unique_months = sorted(df['incident_month'].dropna().unique())
month_options = [calendar.month_name[int(m)] for m in unique_months if 0 < int(m) < 13]
if not month_options:
    month_options = [calendar.month_name[m] for m in range(1, 13)]
selected_month = st.sidebar.selectbox(
    "Select Crime Month",
    options=["All"] + month_options,
    index=0
)
if selected_month == "All":
    selected_months = [i for i in unique_months]
else:
    selected_months = [i for i, name in enumerate(calendar.month_name) if name == selected_month]

# Time of day filter
unique_times = sorted(df['part_of_the_day'].dropna().unique())
selected_time = st.sidebar.selectbox(
    "Select Time of Day",
    options=["All"] + unique_times,
    index=0
)
if selected_time == "All":
    selected_times = unique_times
else:
    selected_times = [selected_time]

# Filter dataframe
filtered_df = df[
    df['incident_division'].isin(selected_divisions) &
    df['crime'].isin(selected_crimes) &
    df['incident_month'].isin(selected_months) &
    df['part_of_the_day'].isin(selected_times)
]

# --- Data Wrangling and EDA Section ---
st.header("1. Data Overview & Exploration")
st.write("A quick look at the raw data and some basic statistics.")

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data Sample")
    st.dataframe(filtered_df.head())

if st.checkbox("Show Data Statistics"):
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered_df.describe())

# --- Interactive Visualizations Section ---
st.header("2. Key Insights and Visualizations")

# Crime Count by Division
st.subheader("Crime Incidents by Division")
division_counts = filtered_df['incident_division'].value_counts().reset_index()
division_counts.columns = ['Division', 'Incident Count']
fig_division = px.bar(
    division_counts,
    x='Division',
    y='Incident Count',
    color='Division',
    title="Total Crime Incidents per Division",
    labels={'Incident Count': 'Number of Incidents'},
    template="plotly_white"
)
st.plotly_chart(fig_division, use_container_width=True)

# Crime Count by Season
st.subheader("Crime Incidents By Season")
season_counts = filtered_df['season'].value_counts().reset_index()
season_counts.columns = ['Season', 'Incident Count']
fig_season = px.pie(
    season_counts,
    names='Season',
    values='Incident Count',
    color='Season',
    title="Total Crime Incidents per Season",
    labels={'Incident Count': 'Number of Incidents'},
    template="plotly_white",
    hole=0.4
)
fig_season.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_season, use_container_width=True)

# Crime Count by Month
st.subheader("Crime Incidents By Month")
month_counts = filtered_df['incident_month'].value_counts().reset_index()
month_counts.columns = ['Month', 'Month Count']
month_counts = month_counts.sort_values('Month')
month_counts['Month Name'] = month_counts['Month'].apply(lambda x: calendar.month_name[int(x)] if 0 < int(x) < 13 else str(x))
fig_month = px.bar(
    month_counts,
    x='Month Name',
    y='Month Count',
    color='Month',
    title="Total Crime Incidents per Month",
    labels={'Incident Count': 'Number of Incidents'},
    template="plotly_white"
)
st.plotly_chart(fig_month, use_container_width=True)

# Crime Count by Weekday
st.subheader("Crime Count by Day of Week")
weekday_counts = filtered_df['incident_weekday'].value_counts().reset_index()
weekday_counts.columns = ['Weekday', 'Count']
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_counts['Weekday'] = pd.Categorical(weekday_counts['Weekday'], categories=weekday_order, ordered=True)
weekday_counts = weekday_counts.sort_values('Weekday')
fig_weekday = px.bar(
    weekday_counts,
    x='Weekday',
    y='Count',
    title="Total Crime Incidents per Day of Week",
    template="plotly_white",
    color='Count'
)
st.plotly_chart(fig_weekday, use_container_width=True)

# Crime Distribution by Part of the Days
st.subheader("Crime Distribution by Time of Day")
time_counts = filtered_df['part_of_the_day'].value_counts().reset_index()
time_counts.columns = ['Part of Day', 'Count']
fig_time = px.pie(
    time_counts,
    values='Count',
    names='Part of Day',
    title="Distribution of Crimes by Time of Day",
    hole=0.4,
)
fig_time.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_time, use_container_width=True)

# Crime Type Horizontal Bar Chart
st.subheader("Crime Count by Type")
crime_type_counts = filtered_df['crime'].value_counts().reset_index()
crime_type_counts.columns = ['Crime Type', 'Count']
fig_crime_type = px.pie(
    crime_type_counts,
    values='Count',
    names='Crime Type',
    title="Total Crime by Type",
    template="plotly_white",
    color='Count'
)
st.plotly_chart(fig_crime_type, use_container_width=True)

# Crime Incidents on the Map
st.subheader("Crime Incidents on the Map")
st.markdown("Zoom and pan to explore crime locations across Bangladesh.")

if not filtered_df.empty:
    map_df = filtered_df.dropna(subset=['latitude', 'longitude', 'crime']).copy()

    map_option = st.radio(
        "Select Map View:",
        options=["Scatter Plot", "Heatmap"],
        horizontal=True
    )

    if map_option == "Scatter Plot":
        fig_map = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            color="crime",
            hover_name="crime",
            zoom=6,
            height=600
        )
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        fig_heatmap = px.density_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            radius=18,
            center={"lat": map_df["latitude"].mean(), "lon": map_df["longitude"].mean()},
            zoom=6,
            height=600,
            color_continuous_scale=["yellow", "orange", "red", "darkred"],
            mapbox_style="carto-positron"
        )
        fig_heatmap.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("No data to display on the map with the current filters.")

st.header("3. Crime Pattern Analysis: Clustering")
st.write(
    "K-Means clustering digunakan untuk menemukan pola kriminal berdasarkan faktor geospasial & populasi."
)

features = ['latitude', 'longitude', 'total_population', 'density_per_kmsq']
clustering_df = df.dropna(subset=features).copy()

if clustering_df.empty:
    st.warning("Data tidak cukup untuk clustering (missing values pada fitur).")
else:
    X = clustering_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow & Silhouette ---
    max_clusters = min(8, len(X_scaled)-1)
    inertia, sil_scores = [], []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = go.Figure(go.Scatter(x=list(range(2, max_clusters+1)), y=inertia, mode="lines+markers"))
        fig_elbow.update_layout(title="Elbow Method", xaxis_title="Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        fig_sil = go.Figure(go.Scatter(x=list(range(2, max_clusters+1)), y=sil_scores, mode="lines+markers"))
        fig_sil.update_layout(title="Silhouette Scores", xaxis_title="Clusters", yaxis_title="Score")
        st.plotly_chart(fig_sil, use_container_width=True)

    # --- User pilih cluster ---
    n_clusters = st.slider("Pilih jumlah cluster", 2, max_clusters, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustering_df["Cluster"] = kmeans.fit_predict(X_scaled)

    # --- PCA ---
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    clustering_df["PCA1"], clustering_df["PCA2"] = coords[:,0], coords[:,1]

    st.subheader(f"Hasil Clustering ({n_clusters} Cluster)")
    fig_pca = px.scatter(clustering_df, x="PCA1", y="PCA2", color="Cluster",
                         hover_data=["incident_division","crime"], template="plotly_white")
    st.plotly_chart(fig_pca, use_container_width=True)

    # --- Map Cluster ---
    fig_map_cluster = px.scatter_mapbox(
        clustering_df, lat="latitude", lon="longitude", color="Cluster",
        hover_name="incident_division", hover_data=["crime","total_population","density_per_kmsq"],
        zoom=6, height=600
    )
    fig_map_cluster.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_map_cluster, use_container_width=True)

    # --- Cluster Highlights ---
    st.subheader("🔎 Key Highlights per Cluster")
    cluster_means = clustering_df.groupby("Cluster")[features].mean()
    st.dataframe(cluster_means.style.background_gradient(cmap="Blues"))

    for cl in sorted(clustering_df["Cluster"].unique()):
        sub = clustering_df[clustering_df["Cluster"]==cl]
        avg_pop = int(sub["total_population"].mean())
        avg_den = round(sub["density_per_kmsq"].mean(),2)
        top_div = sub["incident_division"].value_counts().idxmax()
        top_crime = sub["crime"].value_counts().idxmax()

        st.markdown(
            f"""
            **Cluster {cl}:**
            - Rata-rata populasi: {avg_pop:,}
            - Rata-rata density: {avg_den} orang/km²
            - Divisi dominan: {top_div}
            - Kejahatan paling sering: {top_crime}
            """
        )

st.markdown("---")