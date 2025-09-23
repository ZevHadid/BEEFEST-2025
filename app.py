import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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
fig_crime_type = px.bar(
    crime_type_counts,
    x='Count',
    y='Crime Type',
    orientation='h',
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
        ratio = round(row["total_population"] / row["crime_count"]) if row["crime_count"] > 0 else "-"
        st.markdown(f"<h4>Di {row['incident_division']}, terdapat {row['crime_count']} kasus kriminal tercatat. Itu berarti sekitar 1 dari {ratio} penduduk pernah tercatat melakukan kejahatan.</h4>", unsafe_allow_html=True)

    st.caption("Perhitungan rasio berdasarkan jumlah kasus kriminal dibagi dengan rata-rata populasi di tiap divisi.")
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

# --- Machine Learning Section ---
st.header("3. Machine Learning Model: Crime Type Prediction")
st.write(
    "This section uses a Decision Tree Classifier to predict the type of crime based on various "
    "geographical and environmental factors."
)

ml_df = df.dropna(subset=['latitude', 'longitude', 'weather_code', 'humidity', 'total_population', 'density_per_kmsq', 'crime']).copy()

features = ['latitude', 'longitude', 'weather_code', 'humidity', 'total_population', 'density_per_kmsq']
target = 'crime'

X = ml_df[features]
y = ml_df[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")
st.info("The model's accuracy is a good starting point but can be improved with more features and a more complex model.")

# --- Interactive Prediction Form ---
st.subheader("Predict a Crime Type")
st.sidebar.header("Make a Prediction")
st.sidebar.markdown("Input the values below to get a crime prediction.")

pred_latitude = st.sidebar.slider("Latitude", float(X['latitude'].min()), float(X['latitude'].max()), float(X['latitude'].mean()))
pred_longitude = st.sidebar.slider("Longitude", float(X['longitude'].min()), float(X['longitude'].max()), float(X['longitude'].mean()))
pred_weather_code = st.sidebar.number_input("Weather Code", value=int(X['weather_code'].mean()))
pred_humidity = st.sidebar.slider("Humidity (%)", float(X['humidity'].min()), float(X['humidity'].max()), float(X['humidity'].mean()))
pred_population = st.sidebar.number_input("Total Population", value=int(X['total_population'].mean()), step=1000)
pred_density = st.sidebar.number_input("Density (per sq km)", value=float(X['density_per_kmsq'].mean()))

if st.sidebar.button("Predict Crime"):
    input_data = pd.DataFrame([[pred_latitude, pred_longitude, pred_weather_code, pred_humidity, pred_population, pred_density]],
                              columns=features)
    predicted_encoded = model.predict(input_data)
    predicted_crime = le.inverse_transform(predicted_encoded)
    st.sidebar.success(f"The predicted crime is: **{predicted_crime[0]}**")
    st.sidebar.markdown(f"**How it works:** The model uses the features you provided to classify the most likely crime type.")

st.markdown("---")