import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Spatio-Temporal LSTM Weather Engine",
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
.stAlert { border-radius: 10px; }
h1, h2, h3 { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("Distributed Deep Learning Weather & Pollution Engine")

# UPGRADE: Reflected the new v3 architecture in the sidebar
mode = st.sidebar.radio("Data Engine", ["Real-Time LSTM Stream (v3)", "Global Historical (HDFS)"])

@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("WeatherDashboard_UI").getOrCreate()

# REALTIME STREAM DATA
@st.cache_data(ttl=5)
def load_stream_data():
    spark = get_spark()
    try:
        # UPGRADE: Pointing to the new v3 LSTM Data Lake
        df = spark.read.parquet("hdfs://localhost:9000/weather-data/live_lake_v3")
        pdf = df.toPandas()
        
        cols_to_numeric = ["temperature", "humidity", "predicted_pm2_5", "wind_speed", "latitude", "longitude"]
        for col in cols_to_numeric:
            if col in pdf.columns:
                pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
                
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], errors="coerce")
        pdf = pdf.sort_values("timestamp")
        
        # Recalculating anomaly for the UI so we keep the cool red-alert feature
        pdf["temp_volatility"] = pdf.groupby("city")["temperature"].transform(
            lambda x: x.rolling(window=5, min_periods=2).std()
        ).fillna(0.0)
        pdf["is_anomaly"] = pdf["temp_volatility"] > 1.0
            
        return pdf
    except Exception as e:
        return pd.DataFrame()

# GLOBAL DATASET
@st.cache_data
def load_global_dataset():
    spark = get_spark()
    df = spark.read.csv("data/GlobalWeatherRepository.csv", header=True, inferSchema=True)
    pdf = df.toPandas()
    
    features = pdf[["temperature_celsius", "humidity", "wind_kph"]].dropna()
    model = IsolationForest(contamination=0.02, random_state=42)
    pdf.loc[features.index, "anomaly"] = model.fit_predict(features)
    
    return pdf

# ==========================================
# REAL-TIME DASHBOARD LOGIC
# ==========================================
if mode == "Real-Time LSTM Stream (v3)":
    st_autorefresh(interval=5000, key="data_refresh")
    df = load_stream_data()

    if df.empty:
        st.warning("Waiting for Kafka stream & LSTM processor to write to HDFS v3...")
        st.stop()
    
    st.header("System Health & LSTM Memory Monitor")
    recent_threshold = df["timestamp"].max() - pd.Timedelta(minutes=2)
    recent_anomalies = df[(df["timestamp"] >= recent_threshold) & (df["is_anomaly"] == True)]
    
    if not recent_anomalies.empty:
        affected_cities = ", ".join(recent_anomalies["city"].unique())
        st.error(f"⚠️ **STATUS: ANOMALY DETECTED** | High micro-climate volatility in: {affected_cities}")
    else:
        st.success("**STATUS: GOOD** | Regional micro-climates are currently stable.")
    
    st.divider()

    st.header("Live Pipeline Telemetry")
    total_events = len(df)
    latest_batch_time = df["timestamp"].max()
    one_min_ago = latest_batch_time - pd.Timedelta(seconds=60)
    current_throughput = len(df[df["timestamp"] >= one_min_ago])
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Total Events Processed", f"{total_events:,}")
    col_t2.metric("Current Throughput", f"{current_throughput} events/min")
    col_t3.metric("Latest Micro-batch", latest_batch_time.strftime("%H:%M:%S") if pd.notnull(latest_batch_time) else "N/A")
    col_t4.metric("Deep Learning Engine", "PyTorch LSTM Active")

    st.divider()

    selected_city = st.selectbox("Select City for Live View", ["All"] + sorted(df["city"].unique().tolist()))
    city_df = df[df["city"] == selected_city] if selected_city != "All" else df

    st.header(f"🌤️ Live Insights: {selected_city}")
    latest_data = city_df.groupby("city").last().reset_index()
    
    # UPGRADE: Show the live LSTM Buffer status dynamically
    if "buffer_status" in latest_data.columns:
        b_status = latest_data["buffer_status"].iloc[0] if len(latest_data) == 1 else "Mixed"
        if "Buffering" in b_status:
            st.info(f"**LSTM Memory State:** {b_status} - Waiting for sequence to fill before predicting.")
        else:
            st.success(f"**LSTM Memory State:** {b_status} - Predicting PM2.5 drift.")

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    
    avg_temp = latest_data["temperature"].mean()
    avg_pm25 = latest_data["predicted_pm2_5"].mean()
    avg_hum = latest_data["humidity"].mean()
    avg_vol = latest_data["temp_volatility"].mean()

    with col_g1:
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number", value = avg_temp, title = {'text': "Temperature (°C)"},
            gauge = {'axis': {'range': [-10, 50]}, 'bar': {'color': "#ef4444"}}
        ))
        fig1.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        bar_color = "#10b981" if avg_pm25 < 15 else ("#f59e0b" if avg_pm25 < 35 else "#ef4444")
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number", value = avg_pm25, title = {'text': "Air Quality (PM2.5)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': bar_color}}
        ))
        fig2.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_g3:
        fig3 = go.Figure(go.Indicator(
            mode = "gauge+number", value = avg_hum, title = {'text': "Humidity (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}}
        ))
        fig3.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_g4:
        fig4 = go.Figure(go.Indicator(
            mode = "number+delta", value = avg_vol, title = {'text': "Temp Volatility (σ)"},
            delta = {'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}}
        ))
        fig4.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    st.header("Advanced Sequence Telemetry")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live LSTM PM2.5 Distribution")
        trend_df = city_df.tail(300)
        
        fig_trend = px.scatter(
            trend_df, x="timestamp", y="predicted_pm2_5", color="city", opacity=0.6 
        )
        fig_trend.update_traces(marker=dict(size=8, line=dict(width=1, color="rgba(255,255,255,0.2)")))
        fig_trend.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="WHO Safe Limit")
        fig_trend.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col2:
        st.subheader("Micro-Climate Volatility Trends")
        fig_vol = px.area(trend_df, x="timestamp", y="temp_volatility", color="city")
        fig_vol.add_hline(y=1.0, line_dash="solid", line_color="#ef4444", annotation_text="Anomaly Threshold")
        fig_vol.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Real-Time Native Telemetry Map")
    if "latitude" in latest_data.columns and "longitude" in latest_data.columns:
        latest_data["marker_size"] = latest_data.apply(
            lambda row: 25 if row.get("is_anomaly", False) else (row.get("predicted_pm2_5", 5) + 5), axis=1
        )
        
        fig_map = px.scatter_mapbox(
            latest_data, lat="latitude", lon="longitude", size="marker_size", color="temperature",
            hover_name="city", hover_data=["predicted_pm2_5", "temp_volatility", "is_anomaly", "buffer_status"], 
            zoom=1.5, height=400, color_continuous_scale="Inferno"
        )
        fig_map.update_layout(mapbox_style="carto-darkmatter")
        st.plotly_chart(fig_map, use_container_width=True)

# ==========================================
# GLOBAL DATASET LOGIC
# ==========================================
else:
    df_global = load_global_dataset()
    st.header("Global Historical Analytics")
    st.caption("Deep dive into worldwide meteorological trends.")
    
    df_sample = df_global.sample(n=min(3000, len(df_global)), random_state=42)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records Tracked", len(df_global))
    col2.metric("Countries Analyzed", df_global["country"].nunique())
    col3.metric("Avg Global Temp", f"{round(df_global['temperature_celsius'].mean(),1)} °C")
    col4.metric("Avg Global PM2.5", f"{round(df_global['air_quality_PM2.5'].mean(),1)}")
    st.divider()

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

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Actual vs 'Feels Like' Temperature")
        fig5 = px.scatter(df_sample, x="temperature_celsius", y="feels_like_celsius", color="humidity", opacity=0.6)
        fig5.add_shape(type="line", x0=-20, y0=-20, x1=50, y1=50, line=dict(color="White", dash="dash"))
        st.plotly_chart(fig5, use_container_width=True)
        
    with c6:
        st.subheader("How Humidity Affects Visibility")
        df_sample['Humidity_Zone'] = pd.cut(
            df_sample['humidity'], bins=[0, 30, 60, 85, 100], 
            labels=['Dry (<30%)', 'Moderate (30-60%)', 'Humid (60-85%)', 'Very Humid (>85%)']
        )
        vis_trend = df_sample.groupby('Humidity_Zone')['visibility_km'].mean().reset_index()
        fig6 = px.bar(
            vis_trend, x="Humidity_Zone", y="visibility_km", text_auto='.1f',
            color="visibility_km", color_continuous_scale="Blues_r",
            labels={"visibility_km": "Average Visibility (km)", "Humidity_Zone": "Humidity Level"}
        )
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Wind Speed Distribution")
        fig7 = px.histogram(df_sample, x="wind_kph", nbins=40, color_discrete_sequence=["#10b981"])
        st.plotly_chart(fig7, use_container_width=True)
        
    with c8:
        st.subheader("UV Index by Top Regions")
        top_countries = df_global["country"].value_counts().nlargest(10).index
        box_df = df_global[df_global["country"].isin(top_countries)]
        fig8 = px.box(box_df, x="country", y="uv_index", color="country")
        st.plotly_chart(fig8, use_container_width=True)

    c9, c10 = st.columns(2)
    with c9:
        st.subheader("Global 2D Temperature Map")
        fig9 = px.density_mapbox(
            df_sample, lat="latitude", lon="longitude", z="temperature_celsius", 
            radius=10, zoom=0.5, mapbox_style="carto-darkmatter"
        )
        st.plotly_chart(fig9, use_container_width=True)

    with c10:
        st.subheader("Historical Weather Anomalies")
        fig10 = px.scatter(
            df_sample, x="temperature_celsius", y="humidity", color="anomaly", 
            hover_name="location_name", color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig10, use_container_width=True)