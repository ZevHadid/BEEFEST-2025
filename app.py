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
'''st.subheader("Crime Incidents on the Map")
st.markdown("Zoom and pan to explore crime locations across Bangladesh. The dots represent reported incidents.")
if not filtered_df.empty:
    st.map(filtered_df[['latitude', 'longitude', 'crime']].dropna(), color='crime')
else:
    st.warning("No data to display on the map with the current filters.")'''

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

# --- Machine Learning Section ---
st.header("3. Machine Learning Model: Crime Type Prediction")
st.write(
    "This section uses a Decision Tree Classifier to predict the type of crime based on various "
    "geographical and environmental factors."
)

# Prepare data for the model
# Drop rows with NaN in key features and target
ml_df = df.dropna(subset=['latitude', 'longitude', 'weather_code', 'humidity', 'total_population', 'density_per_kmsq', 'crime']).copy()

# Features and target
features = ['latitude', 'longitude', 'weather_code', 'humidity', 'total_population', 'density_per_kmsq']
target = 'crime'

X = ml_df[features]
y = ml_df[target]

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")
st.info("The model's accuracy is a good starting point but can be improved with more features and a more complex model.")

# --- Interactive Prediction Form ---
st.subheader("Predict a Crime Type")
st.sidebar.header("Make a Prediction")
st.sidebar.markdown("Input the values below to get a crime prediction.")

# Sliders and inputs for user prediction
pred_latitude = st.sidebar.slider("Latitude", float(X['latitude'].min()), float(X['latitude'].max()), float(X['latitude'].mean()))
pred_longitude = st.sidebar.slider("Longitude", float(X['longitude'].min()), float(X['longitude'].max()), float(X['longitude'].mean()))
pred_weather_code = st.sidebar.number_input("Weather Code", value=int(X['weather_code'].mean()))
pred_humidity = st.sidebar.slider("Humidity (%)", float(X['humidity'].min()), float(X['humidity'].max()), float(X['humidity'].mean()))
pred_population = st.sidebar.number_input("Total Population", value=int(X['total_population'].mean()), step=1000)
pred_density = st.sidebar.number_input("Density (per sq km)", value=float(X['density_per_kmsq'].mean()))

# Create a button to trigger prediction
if st.sidebar.button("Predict Crime"):
    # Prepare the input for the model
    input_data = pd.DataFrame([[pred_latitude, pred_longitude, pred_weather_code, pred_humidity, pred_population, pred_density]],
                              columns=features)
    
    # Predict the crime
    predicted_encoded = model.predict(input_data)
    predicted_crime = le.inverse_transform(predicted_encoded)
    
    # Display the result
    st.sidebar.success(f"The predicted crime is: **{predicted_crime[0]}**")
    st.sidebar.markdown(f"**How it works:** The model uses the features you provided to classify the most likely crime type.")

st.markdown("---")

