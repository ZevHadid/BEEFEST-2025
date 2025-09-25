import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import calendar

# --- Main App Configuration ---
st.set_page_config(
    page_title="Bangladesh Crime Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load the Data ---
data_file_path = "Dataset 3 - Bangladesh Crime Dataset.csv"
df = pd.read_csv(data_file_path)

# --- Title and Introduction ---
st.title("Interactive Crime Data Analysis in Bangladesh")
st.markdown("### A data exploration dashboard built with Streamlit.")
st.write(
    "This application provides an interactive platform to analyze crime data in Bangladesh. "
    "Use the sidebar to filter data and interact with the visualizations."
)

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")

# Multi-select for incident division
unique_divisions = sorted(df['incident_division'].unique())
selected_divisions = st.sidebar.multiselect(
    "Select Division(s)",
    options=unique_divisions,
    default=unique_divisions
)

# Multi-select for crime type
unique_crimes = sorted(df['crime'].unique())
selected_crimes = st.sidebar.multiselect(
    "Select Crime Type(s)",
    options=unique_crimes,
    default=unique_crimes
)

# Multi-select for months (show names)
unique_months = sorted(df['incident_month'].unique())
selected_month_names = st.sidebar.multiselect(
    "Select Crime Month(s)",
    options=[calendar.month_name[m] for m in unique_months],
    default=[calendar.month_name[m] for m in unique_months]
)

# Convert selected names back to numbers
selected_months = [i for i, name in enumerate(calendar.month_name) if name in selected_month_names]

# Filter the dataframe based on selections
filtered_df = df[
    df['incident_division'].isin(selected_divisions) &
    df['crime'].isin(selected_crimes) &
    df['incident_month'].isin(selected_months)
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
month_counts['Month Name'] = month_counts['Month'].apply(lambda x: calendar.month_name[x])
fig_month = px.bar(
    month_counts,
    x='Month Name',
    y='Month Count',
    color='Month',
    title="Tital Crime Incidents per Month",
    labels={'Incident Count': 'Number of Incidents'},
    template="plotly_white"
)
st.plotly_chart(fig_month, use_container_width=True)

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

# Crime Incidents on a Map
if not filtered_df.empty:
    fig = px.scatter_map(
        filtered_df.dropna(subset=['latitude', 'longitude', 'crime']),
        lat='latitude',
        lon='longitude',
        color='crime',          
        hover_name='crime',
        zoom=6,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)
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
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)

    # Perform K-means clustering
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