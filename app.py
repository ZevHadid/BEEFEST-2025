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

# Crime Rate Narrative Section
st.subheader("Crime Rate vs Population Narrative")

if not filtered_df.empty:
    rate_df = filtered_df.groupby("incident_division").agg({
        "crime": "count",
        "total_population": "mean"
    }).reset_index().rename(columns={"crime":"crime_count"})

    rate_df["crime_rate_per_1000"] = (rate_df["crime_count"] / rate_df["total_population"]) * 1000

    for _, row in rate_df.iterrows():
        if row["crime_count"] > 0 and row["total_population"] > 0:
            percent = (row["crime_count"] / row["total_population"]) * 100
            st.markdown(
                f"<h4>Di {row['incident_division']}, terdapat {row['crime_count']} kasus kriminal tercatat. "
                f"Itu berarti sekitar {percent:.2f}% dari {int(row['total_population'])} penduduk pernah tercatat melakukan kejahatan.</h4>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h4>Di {row['incident_division']}, tidak ada data yang cukup untuk menghitung persentase kejahatan.</h4>",
                unsafe_allow_html=True
            )

    st.caption("Perhitungan persentase berdasarkan jumlah kasus kriminal dibagi dengan rata-rata populasi di tiap divisi.")

else:
    st.info("Data tidak tersedia untuk membuat narasi rasio.")

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
        fig_map = px.scatter_mapbox(
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

# --- Machine Learning Section - Clustering ---
st.header("3. Crime Pattern Analysis: Clustering")
st.write(
    "This section uses K-Means clustering to identify patterns and groups in crime data based on "
    "geographical, temporal, and environmental factors."
)

# Check available columns and create appropriate features
st.subheader("Available Columns for Clustering")
st.write("Columns in your dataset:", list(df.columns))

# Create a safe list of features that actually exist in the dataset
available_features = []
potential_features = [
    'latitude', 'longitude', 'incident_month', 'weather_code', 
    'humidity', 'total_population', 'density_per_kmsq'
]

for feature in potential_features:
    if feature in df.columns:
        available_features.append(feature)

st.write("Features available for clustering:", available_features)

# Prepare data for clustering with only available features
clustering_df = df.dropna(subset=available_features).copy()

if len(clustering_df) == 0:
    st.warning("No data available for clustering after removing missing values.")
else:
    X = clustering_df[available_features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar for clustering parameters
    st.sidebar.header("Clustering Parameters")
    
    # Elbow Method Section
    st.subheader("Elbow Method for Optimal Cluster Selection")
    st.write("The elbow method helps determine the optimal number of clusters by finding the point where the inertia (within-cluster sum of squares) starts decreasing linearly.")
    
    # Calculate inertia for different numbers of clusters
    max_clusters = min(10, len(X_scaled) - 1)  # Ensure we don't exceed data points
    inertia_values = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    with st.spinner("Calculating optimal number of clusters..."):
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia_values.append(kmeans.inertia_)
            
            # Calculate silhouette score (if enough data points)
            if k < len(X_scaled):
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            else:
                silhouette_scores.append(0)
    
    # Create elbow plot
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range),
        y=inertia_values,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    fig_elbow.update_layout(
        title='Elbow Method for Optimal Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Inertia (Within-cluster Sum of Squares)',
        template="plotly_white"
    )
    
    # Add silhouette score plot
    fig_silhouette = go.Figure()
    fig_silhouette.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    fig_silhouette.update_layout(
        title='Silhouette Scores for Different Numbers of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Silhouette Score',
        template="plotly_white"
    )
    
    # Display both plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    # Find suggested optimal k (elbow point)
    # Simple method: find the point where the decrease in inertia starts to slow down
    differences = [inertia_values[i-1] - inertia_values[i] for i in range(1, len(inertia_values))]
    if differences:
        # Find the point with the largest change in the rate of decrease
        second_differences = [differences[i-1] - differences[i] for i in range(1, len(differences))]
        if second_differences:
            suggested_k = k_range[second_differences.index(max(second_differences)) + 2]
        else:
            suggested_k = 3  # Default fallback
    else:
        suggested_k = 3
    
    # Also consider silhouette scores
    silhouette_optimal = k_range[silhouette_scores.index(max(silhouette_scores))]
    
    st.info(f"""
    **Optimal Cluster Suggestions:**
    - **Elbow method suggests**: {suggested_k} clusters
    - **Silhouette method suggests**: {silhouette_optimal} clusters (score: {max(silhouette_scores):.3f})
    """)
    
    # Let user choose between suggested or manual
    cluster_choice = st.radio(
        "Choose number of clusters:",
        ["Use elbow method suggestion", "Use silhouette method suggestion", "Choose manually"],
        index=0
    )
    
    if cluster_choice == "Use elbow method suggestion":
        n_clusters = suggested_k
    elif cluster_choice == "Use silhouette method suggestion":
        n_clusters = silhouette_optimal
    else:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=max_clusters, value=suggested_k)
    
    st.sidebar.write(f"**Selected clusters:** {n_clusters}")

    # Perform K-means clustering with selected k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    clustering_df['Cluster'] = clusters

    # Add PCA for visualization if we have enough features
    if len(available_features) > 1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        clustering_df['PCA1'] = X_pca[:, 0]
        clustering_df['PCA2'] = X_pca[:, 1]
        
        # Display PCA explained variance
        st.write(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.2f}")

    # Display clustering results
    st.subheader(f"Clustering Results ({n_clusters} Clusters)")

    # Cluster distribution
    cluster_counts = clustering_df['Cluster'].value_counts().sort_index()
    fig_cluster_dist = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Incidents'},
        title=f"Distribution of Crime Incidents Across {n_clusters} Clusters",
        color=cluster_counts.index,
        template="plotly_white"
    )
    st.plotly_chart(fig_cluster_dist, use_container_width=True)

    # PCA visualization (only if we have PCA components)
    if 'PCA1' in clustering_df.columns and 'PCA2' in clustering_df.columns:
        fig_pca = px.scatter(
            clustering_df,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            hover_data=['crime', 'incident_division'],
            title="PCA Visualization of Crime Clusters",
            template="plotly_white"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # Cluster characteristics
    st.subheader("Cluster Characteristics")

    # Display mean values for each cluster
    cluster_means = clustering_df.groupby('Cluster')[available_features].mean()
    st.write("Average Feature Values by Cluster:")
    st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))

    # Crime type distribution within clusters
    st.subheader("Crime Type Distribution within Clusters")
    cluster_crime_dist = pd.crosstab(clustering_df['Cluster'], clustering_df['crime'], normalize='index') * 100
    fig_cluster_crime = px.imshow(
        cluster_crime_dist,
        title="Crime Type Distribution by Cluster (%)",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_cluster_crime, use_container_width=True)

    # Geographical visualization of clusters
    st.subheader("Geographical Distribution of Clusters")
    if not clustering_df.empty:
        fig_cluster_map = px.scatter_map(
            clustering_df,
            lat='latitude',
            lon='longitude',
            color='Cluster',
            hover_name='crime',
            hover_data=['incident_division', 'incident_month'],
            title="Geographical Distribution of Crime Clusters",
            zoom=6,
            height=600
        )
        fig_cluster_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_cluster_map, use_container_width=True)

    # Cluster interpretation
    st.subheader("Cluster Interpretation")
    st.write("""
    **How to interpret the clusters:**
    - Each cluster represents a group of crime incidents with similar characteristics
    - Clusters may represent patterns like: urban vs rural crimes, seasonal patterns, 
      weather-related patterns, or population density correlations
    - Use the cluster characteristics table to understand what makes each cluster unique
    """)

    # Interactive cluster exploration
    st.sidebar.header("Explore Specific Cluster")
    selected_cluster = st.sidebar.selectbox("Select Cluster to Explore", options=sorted(clustering_df['Cluster'].unique()))

    if selected_cluster is not None:
        cluster_data = clustering_df[clustering_df['Cluster'] == selected_cluster]
        
        st.subheader(f"Detailed Analysis of Cluster {selected_cluster}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top crimes in cluster
            top_crimes = cluster_data['crime'].value_counts().head(5)
            fig_top_crimes = px.bar(
                x=top_crimes.index,
                y=top_crimes.values,
                title=f"Top 5 Crimes in Cluster {selected_cluster}",
                labels={'x': 'Crime Type', 'y': 'Count'},
                color=top_crimes.values,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_top_crimes, use_container_width=True)
        
        with col2:
            # Division distribution in cluster
            division_dist = cluster_data['incident_division'].value_counts()
            fig_division_dist = px.pie(
                values=division_dist.values,
                names=division_dist.index,
                title=f"Division Distribution in Cluster {selected_cluster}",
                hole=0.4
            )
            st.plotly_chart(fig_division_dist, use_container_width=True)

st.markdown("---")