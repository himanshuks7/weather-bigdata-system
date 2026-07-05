import os
import json
import warnings
import pandas as pd
import numpy as np
import torch
import joblib
from typing import Iterable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import *
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. MODULE-LEVEL CACHE FOR ML ASSETS
model_cache = {}

def get_models():
    if "model" not in model_cache:
        model_cache["model"] = torch.jit.load("../ml/pm25_lstm_traced.pt")
        model_cache["scaler"] = joblib.load("../ml/lstm_scaler.pkl")
    return model_cache["model"], model_cache["scaler"]

# 2. SPARK SESSION & SCHEMAS
spark = SparkSession.builder \
    .appName("Weather_AI_LSTM_Stateful_Processor") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.adaptive.enabled", "false") \
    .getOrCreate()

# THE FIX: Upgraded log level from WARN to ERROR to hide Spark chatter
spark.sparkContext.setLogLevel("ERROR")

print("PySpark Stateful LSTM Sequence Engine Initialized")

weather_schema = StructType([
    StructField("city", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("pressure", DoubleType()),
    StructField("wind_speed", DoubleType()),
    StructField("cloud", DoubleType()),
    StructField("weather", StringType()),
    StructField("timestamp", StringType()),
    StructField("month", IntegerType()),
    StructField("hour", IntegerType())
])

state_schema = StructType([StructField("buffer_json", StringType())])

out_schema = StructType([
    StructField("city", StringType()),
    StructField("timestamp", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("pressure", DoubleType()),
    StructField("wind_speed", DoubleType()),
    StructField("cloud", DoubleType()),
    StructField("predicted_pm2_5", DoubleType()),
    StructField("buffer_status", StringType())
])

# 3. THE STATEFUL DEEP LEARNING UDF
def process_sequence(key: tuple, pdfs: Iterable[pd.DataFrame], state: GroupState) -> Iterable[pd.DataFrame]:
    city = key[0]

    # A. Load Historical State 
    if state.exists:
        state_data = json.loads(state.get[0])
        buffer_df = pd.DataFrame(state_data)
    else:
        buffer_df = pd.DataFrame()

    model, scaler = get_models()
    
    training_feature_cols = [
        "latitude", "longitude", "month", "hour", 
        "temperature_celsius", "humidity", "pressure_mb", 
        "wind_kph", "cloud"
    ]
    
    rename_map = {
        "temperature": "temperature_celsius",
        "pressure": "pressure_mb",
        "wind_speed": "wind_kph"
    }

    # B. Process Micro-Batch Events
    for pdf in pdfs:
        pdf = pdf.sort_values("timestamp")
        results = []

        for _, row in pdf.iterrows():
            new_row_df = pd.DataFrame([row])
            
            # Clean concat logic to avoid Pandas FutureWarnings
            if buffer_df.empty:
                buffer_df = new_row_df
            else:
                buffer_df = pd.concat([buffer_df, new_row_df]).reset_index(drop=True)

            # Slide window
            if len(buffer_df) > 10:
                buffer_df = buffer_df.tail(10)

            # C. Deep Learning Inference
            if len(buffer_df) == 10:
                scaler_df = buffer_df.rename(columns=rename_map)
                X_scaled = scaler.transform(scaler_df[training_feature_cols])
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0) 
                
                with torch.no_grad():
                    pred = max(0.0, model(X_tensor).item()) 
                status = "Active LSTM Drift Inference"
            else:
                pred = 0.0
                status = f"Buffering Sequence ({len(buffer_df)}/10)"

            out_dict = row.to_dict()
            out_dict["predicted_pm2_5"] = float(pred)
            out_dict["buffer_status"] = status
            results.append(out_dict)

        # D. Save updated memory back to Spark Node
        state.update((buffer_df.to_json(orient="records"),))

        yield pd.DataFrame(results)[["city", "timestamp", "latitude", "longitude", "temperature", "humidity", "pressure", "wind_speed", "cloud", "predicted_pm2_5", "buffer_status"]]

# 4. STREAM EXECUTION

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "weather_stream_v3") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), weather_schema).alias("data")) \
    .select("data.*")

# THE SHUFFLE BARRIER: Prevents Kafka offset NPEs by breaking direct lineage
lstm_stream = parsed_df \
    .repartition("city") \
    .groupBy("city") \
    .applyInPandasWithState(
        process_sequence,
        outputStructType=out_schema,
        stateStructType=state_schema,
        outputMode="update",
        timeoutConf=GroupStateTimeout.NoTimeout
    )

def process_microbatch(batch_df, batch_id):
    print(f"\n--- LSTM Micro-Batch {batch_id} ---")
    batch_df.show(20, truncate=False)
    
    if not batch_df.isEmpty():
        batch_df.write \
            .mode("append") \
            .parquet("hdfs://localhost:9000/weather-data/live_lake_v3")

lstm_query = lstm_stream.writeStream \
    .outputMode("update") \
    .foreachBatch(process_microbatch) \
    .option("checkpointLocation", "hdfs://localhost:9000/weather-data/checkpoints/live_lake_v3") \
    .trigger(processingTime="15 seconds") \
    .start()

spark.streams.awaitAnyTermination()