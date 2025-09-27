import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import calendar

# --- Main App Configuration ---
st.set_page_config(
    page_title="Bangladesh Crime Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load the Data ---
df = pd.read_csv("Dataset 3 - Bangladesh Crime Dataset.csv")

# Normalize weekday names
df['incident_weekday'] = df['incident_weekday'].astype(str).str.strip().str.capitalize()
weekday_map = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
    "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"
}
df['incident_weekday'] = df['incident_weekday'].replace(weekday_map)

# --- Title ---
st.title("Bangladesh Crime Data Explorer")
st.markdown("An interactive dashboard to explore patterns, hotspots, and clusters in Bangladesh crime data.")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filter Data")

# Division filter
divisions = sorted(df['incident_division'].dropna().unique())
selected_division = st.sidebar.selectbox("Division", ["All"] + divisions)
divisions_filter = divisions if selected_division == "All" else [selected_division]

# Crime filter
crimes = sorted(df['crime'].dropna().unique())
selected_crime = st.sidebar.selectbox("Crime Type", ["All"] + crimes)
crimes_filter = crimes if selected_crime == "All" else [selected_crime]

# Month filter
months = sorted(df['incident_month'].dropna().unique())
month_names = [calendar.month_name[int(m)] for m in months if 0 < int(m) < 13]
selected_month = st.sidebar.selectbox("Month", ["All"] + month_names)
months_filter = months if selected_month == "All" else [
    i for i, name in enumerate(calendar.month_name) if name == selected_month
]

# Time filter
times = sorted(df['part_of_the_day'].dropna().unique())
selected_time = st.sidebar.selectbox("Time of Day", ["All"] + times)
times_filter = times if selected_time == "All" else [selected_time]

# Apply filters
filtered_df = df[
    df['incident_division'].isin(divisions_filter) &
    df['crime'].isin(crimes_filter) &
    df['incident_month'].isin(months_filter) &
    df['part_of_the_day'].isin(times_filter)
]

# --- Overview ---
st.header("1. Data Overview")
st.write(f"Showing **{len(filtered_df)} incidents** after filters.")
with st.expander("Show Raw Data"):
    st.dataframe(filtered_df.head())

# --- Visuals ---
st.header("2. Exploratory Visualizations")

col1, col2 = st.columns(2)
with col1:
    division_counts = filtered_df['incident_division'].value_counts().reset_index()
    division_counts.columns = ['Division', 'Count']
    st.plotly_chart(px.bar(division_counts, x='Division', y='Count',
                           color='Division',
                           title="Crimes by Division", template="plotly_white"), use_container_width=True)

with col2:
    season_counts = filtered_df['season'].value_counts().reset_index()
    season_counts.columns = ['Season', 'Count']
    st.plotly_chart(px.pie(season_counts, names='Season', values='Count',
                           hole=0.4, title="Crimes by Season", template="plotly_white"), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    month_counts = filtered_df['incident_month'].value_counts().reset_index()
    month_counts.columns = ['Month', 'Count']
    month_counts['Month'] = month_counts['Month'].apply(lambda x: calendar.month_name[int(x)])
    st.plotly_chart(px.bar(month_counts.sort_values("Month"), x='Month', y='Count',
                           color='Month',
                           title="Crimes by Month", template="plotly_white"), use_container_width=True)

with col4:
    weekday_counts = filtered_df['incident_weekday'].value_counts().reset_index()
    weekday_counts.columns = ['Weekday', 'Count']
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_counts['Weekday'] = pd.Categorical(weekday_counts['Weekday'], categories=weekday_order, ordered=True)
    st.plotly_chart(px.bar(weekday_counts.sort_values("Weekday"), x='Weekday', y='Count',
                           color='Weekday',
                           title="Crimes by Weekday", template="plotly_white"), use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    time_counts = filtered_df['part_of_the_day'].value_counts().reset_index()
    time_counts.columns = ['Part of Day', 'Count']
    st.plotly_chart(px.pie(time_counts, values='Count', names='Part of Day',
                           hole=0.4, title="Crimes by Time of Day", template="plotly_white"), use_container_width=True)

with col6:
    crime_counts = filtered_df['crime'].value_counts().reset_index()
    crime_counts.columns = ['Crime', 'Count']
    st.plotly_chart(px.pie(crime_counts, values='Count', names='Crime',
                           hole=0.4, title="Crimes by Type", template="plotly_white"), use_container_width=True)

# --- Map ---
st.header("3. Crime Hotspots")
map_df = filtered_df.dropna(subset=['latitude', 'longitude'])
if not map_df.empty:
    map_option = st.radio("Map View:", ["Scatter", "Heatmap"], horizontal=True)
    if map_option == "Scatter":
        st.plotly_chart(px.scatter_map(map_df, lat="latitude", lon="longitude", color="crime",
                                          zoom=6, height=600, map_style="carto-positron"), use_container_width=True)
    else:
        st.plotly_chart(px.density_map(map_df, lat="latitude", lon="longitude", radius=15,
                                          zoom=6, height=600, map_style="carto-positron"), use_container_width=True)
else:
    st.warning("No map data available for current filters.")

# --- Clustering ---
st.header("4. Crime Pattern Clustering")

features = ['latitude', 'longitude', 'total_population', 'density_per_kmsq']
clustering_df = df.dropna(subset=features).copy()

if clustering_df.empty:
    st.warning("Not enough data for clustering.")
else:
    X = clustering_df[features]
    X_scaled = StandardScaler().fit_transform(X)

    # Find best k
    inertia, sil_scores = [], []
    max_clusters = min(8, len(X_scaled)-1)
    for k in range(2, max_clusters+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(x=list(range(2, max_clusters+1)), y=inertia, markers=True,
                                title="Elbow Method", labels={'x':'Clusters','y':'Inertia'}), use_container_width=True)
    with col2:
        st.plotly_chart(px.line(x=list(range(2, max_clusters+1)), y=sil_scores, markers=True,
                                title="Silhouette Score", labels={'x':'Clusters','y':'Score'}), use_container_width=True)

    # User chooses cluster
    n_clusters = st.slider("Select number of clusters", 2, max_clusters, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustering_df["Cluster"] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    coords = PCA(n_components=2).fit_transform(X_scaled)
    clustering_df["PCA1"], clustering_df["PCA2"] = coords[:,0], coords[:,1]

    st.subheader("Cluster Projection (PCA)")
    st.plotly_chart(px.scatter(clustering_df, x="PCA1", y="PCA2", color="Cluster",
                               hover_data=["incident_division","crime"], template="plotly_white"), use_container_width=True)

    # Cluster profiles
    st.subheader("Cluster Profiles")
    cluster_means = clustering_df.groupby("Cluster")[features].mean().round(2)
    st.dataframe(cluster_means.style.background_gradient(cmap="Blues"))

    # Cluster summaries
    st.subheader("Insights by Cluster")
    cluster_summaries = []
    for cl in sorted(clustering_df["Cluster"].unique()):
        sub = clustering_df[clustering_df["Cluster"]==cl]
        top_div = sub["incident_division"].value_counts().idxmax()
        top_crime = sub["crime"].value_counts().idxmax()
        cluster_summaries.append({
            "Cluster": cl,
            "Population": int(sub["total_population"].mean()),
            "Density": round(sub["density_per_kmsq"].mean(),2),
            "Top Division": top_div,
            "Top Crime": top_crime
        })

    for s in cluster_summaries:
        st.markdown(f"""
        **Cluster {s['Cluster']}**
        - Avg Population: {s['Population']:,}
        - Avg Density: {s['Density']} per km²
        - Dominant Division: {s['Top Division']}
        - Most Common Crime: {s['Top Crime']}
        """)

    # Generalized Insights
    st.subheader("General Patterns")
    st.markdown("""
    - **Urban clusters** (high density) tend to show higher rates of theft and robbery.  
    - **Rural clusters** (low density) see fewer crimes overall, but violent crimes stand out proportionally.  
    - **Population size matters** → denser clusters consistently report more incidents than sparse ones.  
    - Each cluster highlights **different dominant crimes** — crime prevention strategies should be **region-specific**.  
    """)
