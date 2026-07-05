import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Real-Time Distibuted Weather Data Analytics and Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { background: linear-gradient(135deg, #020617, #0f172a); color: white; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}
h1, h2, h3 { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("Real-Time Distibuted Weather Data Analytics and Prediction System")

mode = st.sidebar.radio("Data Engine", ["Real-Time Stream (Kafka/Spark)", "Global Historical (HDFS)"])

@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("WeatherDashboard_UI").getOrCreate()

# REALTIME STREAM DATA
@st.cache_data(ttl=5)
def load_stream_data():
    spark = get_spark()
    try:
        df = spark.read.parquet("hdfs://localhost:9000/weather-data/live_lake")
        pdf = df.toPandas()
        cols_to_numeric = ["temperature", "humidity", "predicted_pm2_5", "wind_speed"]
        for col in cols_to_numeric:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
        return pdf.sort_values("timestamp")
    except Exception as e:
        return pd.DataFrame()

# GLOBAL DATASET
@st.cache_data
def load_global_dataset():
    spark = get_spark()
    df = spark.read.csv("../data/GlobalWeatherRepository.csv", header=True, inferSchema=True)
    pdf = df.toPandas()
    
    # Pre-calculate an anomaly column so it doesn't recalculate on every UI interaction
    features = pdf[["temperature_celsius", "humidity", "wind_kph"]].dropna()
    model = IsolationForest(contamination=0.02, random_state=42)
    pdf.loc[features.index, "anomaly"] = model.fit_predict(features)
    
    return pdf

# REAL-TIME DASHBOARD LOGIC
# **************************
if mode == "Real-Time Stream (Kafka/Spark)":
    # Refresh the dashboard every 5 seconds to show true real-time flow
    st_autorefresh(interval=5000, key="data_refresh")
    df = load_stream_data()

    if df.empty:
        st.warning("Waiting for Kafka stream & Spark processor to write to HDFS...")
        st.stop()

    # Ensure timestamp is a datetime object for calculations
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # --- 0. PIPELINE TELEMETRY (The "Proof of Streaming" Section) ---
    st.header("⚡ Live Pipeline Telemetry")
    st.caption("Monitoring Kafka Ingestion and Spark Structured Streaming Micro-batches")
    
    # Calculate real-time metrics
    total_events = len(df)
    latest_batch_time = df["timestamp"].max()
    
    # Calculate throughput (Events processed in the last 60 seconds)
    one_min_ago = latest_batch_time - pd.Timedelta(seconds=60)
    current_throughput = len(df[df["timestamp"] >= one_min_ago])
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Total Events Ingested", f"{total_events:,}")
    col_t2.metric("Current Throughput", f"{current_throughput} events/min")
    col_t3.metric("Latest Micro-batch", latest_batch_time.strftime("%H:%M:%S") if pd.notnull(latest_batch_time) else "N/A")
    col_t4.metric("Data Lake Status", "Active Writing")

    # Live Event Log (Terminal-style view of the raw data arriving)
    with st.expander("🔴 View Live Raw Kafka/Spark Event Stream", expanded=False):
        st.markdown("<small><i>Showing the 5 most recent events written to HDFS Parquet...</i></small>", unsafe_allow_html=True)
        # Show the most recent 5 records, styled cleanly
        live_feed_df = df.sort_values("timestamp", ascending=False).head(5)
        st.dataframe(
            live_feed_df[["timestamp", "city", "temperature", "humidity", "predicted_pm2_5"]], 
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    selected_city = st.selectbox("Select City for Live View", ["All"] + sorted(df["city"].unique().tolist()))
    city_df = df[df["city"] == selected_city] if selected_city != "All" else df

    st.divider()
    
    # --- 1. THE "NORMAL PERSON" VIEW (Simple & Visual) ---
    st.header("🌤️ Current Weather Snapshot")
    st.caption("Live metrics for the selected region.")
    
    latest_data = city_df.groupby("city").last().reset_index()
    
    # Simple Gauges
    col_g1, col_g2, col_g3 = st.columns(3)
    
    avg_temp = latest_data["temperature"].mean()
    avg_pm25 = latest_data["predicted_pm2_5"].mean()
    avg_hum = latest_data["humidity"].mean()

    with col_g1:
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_temp,
            title = {'text': "Temperature (°C)"},
            gauge = {'axis': {'range': [-10, 50]}, 'bar': {'color': "#ef4444"}}
        ))
        fig1.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_pm25,
            title = {'text': "Air Quality (PM2.5)"},
            gauge = {'axis': {'range': [0, 200]}, 'bar': {'color': "#8b5cf6"}}
        ))
        fig2.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_g3:
        fig3 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_hum,
            title = {'text': "Humidity (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}}
        ))
        fig3.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # --- 2. ADVANCED TELEMETRY (For Data Analysts) ---
    st.header("Advanced Telemetry")
    with st.expander("View Deep Analytics", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Live PM2.5 Analytics Engine")
            trend_df = city_df.tail(300)
            fig_trend = px.line(trend_df, x="timestamp", y="predicted_pm2_5", color="city", markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col2:
            st.subheader("Temperature by City")
            fig_bar = px.bar(
                latest_data, x="city", y="temperature", color="temperature",
                color_continuous_scale="Turbo"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Real-Time Telemetry Map")
        city_coords = {
            "Delhi":[28.6139,77.2090], "Mumbai":[19.0760,72.8777], "Chennai":[13.0827,80.2707], 
            "Bangalore":[12.9716,77.5946], "Kolkata":[22.5726,88.3639], "Hyderabad":[17.3850,78.4867],
            "London":[51.5072,-0.1276], "New York":[40.7128,-74.0060], "Tokyo":[35.6762,139.6503]
        }
        latest_data["lat"] = latest_data["city"].apply(lambda x: city_coords.get(x, [0,0])[0])
        latest_data["lon"] = latest_data["city"].apply(lambda x: city_coords.get(x, [0,0])[1])

        fig_map = px.scatter_mapbox(
            latest_data, lat="lat", lon="lon", size="predicted_pm2_5", color="temperature",
            hover_name="city", zoom=1.5, height=400, color_continuous_scale="Inferno"
        )
        fig_map.update_layout(mapbox_style="carto-darkmatter")
        st.plotly_chart(fig_map, use_container_width=True)


# GLOBAL DATASET LOGIC
#**********************
else:
    df_global = load_global_dataset()
    st.header("🌍 Global Historical Analytics")
    st.caption("Deep dive into worldwide meteorological trends.")
    
    # Fast Sampling for scatter plots to prevent browser crash
    df_sample = df_global.sample(n=min(3000, len(df_global)), random_state=42)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records Tracked", len(df_global))
    col2.metric("Countries Analyzed", df_global["country"].nunique())
    col3.metric("Avg Global Temp", f"{round(df_global['temperature_celsius'].mean(),1)} °C")
    col4.metric("Avg Global PM2.5", f"{round(df_global['air_quality_PM2.5'].mean(),1)}")
    st.divider()

    # --- ROW 1: Extremes ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Hottest Countries")
        hot_df = df_global.groupby("country")["temperature_celsius"].mean().nlargest(10).reset_index()
        fig1 = px.bar(hot_df, x="temperature_celsius", y="country", orientation='h', color="temperature_celsius", color_continuous_scale="Reds")
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        st.subheader("Top 10 Coldest Countries")
        cold_df = df_global.groupby("country")["temperature_celsius"].mean().nsmallest(10).reset_index()
        fig2 = px.bar(cold_df, x="temperature_celsius", y="country", orientation='h', color="temperature_celsius", color_continuous_scale="Blues_r")
        st.plotly_chart(fig2, use_container_width=True)

    # --- ROW 2: Air Quality ---
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Worst Air Quality (PM2.5 Hotspots)")
        pm_df = df_global.groupby("location_name")["air_quality_PM2.5"].mean().nlargest(15).reset_index()
        fig3 = px.bar(pm_df, x="location_name", y="air_quality_PM2.5", color="air_quality_PM2.5", color_continuous_scale="Purples")
        st.plotly_chart(fig3, use_container_width=True)
        
    with c4:
        st.subheader("Weather Conditions Breakdown")
        cond_df = df_global["condition_text"].value_counts().nlargest(8).reset_index()
        cond_df.columns = ["Condition", "Count"]
        fig4 = px.pie(cond_df, names="Condition", values="Count", hole=0.4)
        st.plotly_chart(fig4, use_container_width=True)

    # --- ROW 3: Correlations (Using Sampled Data for Speed) ---
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Actual vs 'Feels Like' Temperature")
        fig5 = px.scatter(df_sample, x="temperature_celsius", y="feels_like_celsius", color="humidity", opacity=0.6)
        fig5.add_shape(type="line", x0=-20, y0=-20, x1=50, y1=50, line=dict(color="White", dash="dash")) # Perfect 1:1 line
        st.plotly_chart(fig5, use_container_width=True)
        
    with c6:
        st.subheader("How Humidity Affects Visibility")
        
        # 1. Group humidity into easy-to-understand zones
        df_sample['Humidity_Zone'] = pd.cut(
            df_sample['humidity'], 
            bins=[0, 30, 60, 85, 100], 
            labels=['Dry (<30%)', 'Moderate (30-60%)', 'Humid (60-85%)', 'Very Humid (>85%)']
        )
        
        # 2. Calculate the average visibility for each zone
        vis_trend = df_sample.groupby('Humidity_Zone')['visibility_km'].mean().reset_index()
        
        # 3. Plot a simple, clean bar chart
        fig6 = px.bar(
            vis_trend, 
            x="Humidity_Zone", 
            y="visibility_km", 
            text_auto='.1f', # Shows the exact number on top of the bar
            color="visibility_km", 
            color_continuous_scale="Blues_r",
            labels={
                "visibility_km": "Average Visibility (km)", 
                "Humidity_Zone": "Humidity Level"
            }
        )
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    # --- ROW 4: Distributions ---
    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Wind Speed Distribution")
        fig7 = px.histogram(df_sample, x="wind_kph", nbins=40, color_discrete_sequence=["#10b981"])
        st.plotly_chart(fig7, use_container_width=True)
        
    with c8:
        st.subheader("UV Index by Top Regions")
        # Get top 10 most frequent countries to keep box plot readable
        top_countries = df_global["country"].value_counts().nlargest(10).index
        box_df = df_global[df_global["country"].isin(top_countries)]
        fig8 = px.box(box_df, x="country", y="uv_index", color="country")
        st.plotly_chart(fig8, use_container_width=True)

    # --- ROW 5: Geospatial and ML ---
    c9, c10 = st.columns(2)
    with c9:
        st.subheader("Global 2D Temperature Map")
        fig9 = px.density_mapbox(
            df_sample, lat="latitude", lon="longitude", z="temperature_celsius", 
            radius=10, zoom=0.5, mapbox_style="carto-darkmatter"
        )
        st.plotly_chart(fig9, use_container_width=True)

    with c10:
        st.subheader("AI Weather Anomalies")
        fig10 = px.scatter(
            df_sample, x="temperature_celsius", y="humidity", color="anomaly", 
            hover_name="location_name", color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig10, use_container_width=True)