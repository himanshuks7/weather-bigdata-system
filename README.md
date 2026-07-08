Weather Big Data System (v3)
A real-time weather and air-quality analytics pipeline built with Kafka, Spark Structured Streaming, PyTorch LSTM, HDFS, and Streamlit.

This README focuses on the v3 workflow and the files you requested:

data_collector/weather_producer.py
ml/train_lstm.py (assuming this is what you meant by train_stm.py)
v3 files (spark/spark_stream_processorv3.py, dashboard/appv3.py)
data/ folder
1) Project Goal
The project simulates and ingests live weather events from multiple cities, performs stateful sequence inference for PM2.5 prediction using an LSTM model, stores enriched stream outputs in HDFS Parquet, and visualizes results in an interactive dashboard.

2) v3 Data Flow (End-to-End)
weather_producer.py fetches weather from OpenWeatherMap for selected cities.
Producer sends enriched JSON events to Kafka topic weather_stream_v3.
spark_stream_processorv3.py consumes Kafka stream.
Spark groups events by city and keeps a rolling sequence buffer (length 10) using applyInPandasWithState.
Once buffer reaches 10 events, Spark applies scaler + traced PyTorch LSTM model to predict PM2.5 drift.
Results are written to HDFS path:
hdfs://localhost:9000/weather-data/live_lake_v3
appv3.py reads HDFS data and renders:
live telemetry
anomaly indicators
PM2.5 sequence trends
spatio-temporal map
global historical analytics from CSV
3) Relevant Files
Data Producer
data_collector/weather_producer.py
Purpose:

Pulls live weather by city from OpenWeatherMap.
Adds temporal and spatial features (month, hour, latitude, longitude).
Emits high-velocity synthetic micro-variations (20 events/city per cycle).
Key details:

Kafka topic: weather_stream_v3
Broker: localhost:9092
Batch loop: all cities, then sleep 30 seconds
Fields emitted:
city, latitude, longitude, temperature, humidity, pressure, wind_speed, cloud, weather, timestamp, month, hour
Model Training
ml/train_lstm.py
Purpose:

Trains a sequence LSTM model to predict air_quality_PM2_5 from historical weather + spatio-temporal features.
Exports artifacts used by streaming inference.
Training input:

../data/GlobalWeatherRepository.csv
Generated artifacts:

../ml/lstm_scaler.pkl (StandardScaler)
../ml/pm25_lstm_traced.pt (TorchScript traced model)
Sequence configuration:

Window length: 10
Features:
latitude, longitude, month, hour, temperature_celsius, humidity, pressure_mb, wind_kph, cloud
Streaming Processor (v3)
spark/spark_stream_processorv3.py
Purpose:

Stateful Spark streaming inference using city-wise rolling sequences.
Loads TorchScript model + scaler once via module-level cache.
Input source:

Kafka topic weather_stream_v3
State logic:

Per-city buffer maintained in Spark state store.
If buffer size < 10: status is buffering and prediction is 0.0.
If buffer size = 10: run scaler + LSTM prediction and output predicted_pm2_5.
Output sink:

Appends Parquet to:
hdfs://localhost:9000/weather-data/live_lake_v3
Checkpoint:
hdfs://localhost:9000/weather-data/checkpoints/live_lake_v3
Dashboard (v3)
dashboard/appv3.py
Purpose:

Streamlit UI for real-time and historical analytics.
Modes:

Real-Time LSTM Stream (v3):
Reads HDFS parquet from live_lake_v3
Computes rolling temp volatility and anomaly flags
Shows throughput, gauges, trends, and map
Global Historical (HDFS):
Reads data/GlobalWeatherRepository.csv
Displays global climate and air-quality visual analytics
4) Data Folder
data/GlobalWeatherRepository.csv
Main historical dataset used for:

LSTM training (ml/train_lstm.py)
Historical dashboard analytics (dashboard/appv3.py)
Expected columns referenced by code include:

Time/location: last_updated, location_name, country, latitude, longitude
Weather: temperature_celsius, feels_like_celsius, humidity, pressure_mb, wind_kph, cloud, visibility_km, uv_index, condition_text
Air quality target: air_quality_PM2_5
5) Prerequisites
Install and run the following services locally:

Python 3.9+
Apache Kafka + Zookeeper
Apache Spark (with PySpark)
Hadoop HDFS (reachable at hdfs://localhost:9000)
Python libraries needed by v3 pipeline:

pandas, numpy, requests, joblib
torch, scikit-learn
pyspark
kafka-python
streamlit, plotly, streamlit-autorefresh
Example installation:

pip install pandas numpy requests joblib torch scikit-learn pyspark kafka-python streamlit plotly streamlit-autorefresh
6) How To Run (Recommended Order)
Run each step in a separate terminal.

Step 1: Start infrastructure
Start Zookeeper
Start Kafka broker (localhost:9092)
Ensure HDFS NameNode/DataNode are up (hdfs://localhost:9000)
Step 2: Train or refresh LSTM artifacts (optional if artifacts already exist)
Important: run from ml/ so relative paths in code resolve correctly.

cd ml
python train_lstm.py
This produces:

ml/lstm_scaler.pkl
ml/pm25_lstm_traced.pt
Step 3: Start Spark streaming processor v3
Important: run from spark/ so ../ml/... artifact paths resolve correctly.

cd spark
python spark_stream_processorv3.py
Step 4: Start weather producer
From project root:

python data_collector/weather_producer.py
Step 5: Launch dashboard
From project root:

streamlit run dashboard/appv3.py
7) Security and Configuration Notes
weather_producer.py currently hardcodes OpenWeatherMap API key (API_KEY).
Move secrets to environment variables in production:
Example: OPENWEATHER_API_KEY
Kafka/HDFS endpoints are currently local defaults.
For cluster deployment, externalize these configs into environment variables or a config file.
8) Operational Tips
If dashboard shows no live data:
verify producer is publishing to weather_stream_v3
verify Spark job is running and writing parquet to HDFS
verify HDFS path exists and is readable
If Spark cannot find model/scaler:
confirm artifacts exist in ml/
run Spark processor from spark/ directory
If timestamp parsing issues appear in dashboard:
check incoming event timestamp format is ISO-8601 (producer already uses UTC ISO format)
9) Known File Name Clarification
You requested train_stm.py; the repository file is currently ml/train_lstm.py. This README documents train_lstm.py.

10) Future Improvements
Replace hardcoded API key and service URLs with .env support.
Add unit tests for sequence logic in process_sequence.
Add Docker Compose for reproducible local startup (Kafka, Spark, HDFS, dashboard).
Add schema validation and dead-letter handling for malformed Kafka messages.
