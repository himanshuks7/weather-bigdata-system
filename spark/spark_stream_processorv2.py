from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, window, avg, count, first, when, stddev
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml import PipelineModel

# Initialize Spark 4.1.1 Session for Structured Streaming
spark = SparkSession.builder \
    .appName("Weather_AI_Stream_Processor_Phase2") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("⚡ Spark Structured Streaming Initialized with Spatio-Temporal and Anomaly Detection.")

# Load the newly trained Gradient-Boosted model
model = PipelineModel.load("../ml/pm25_gbt_model")

# Define exact schema for deterministic parsing 
weather_schema = StructType([
    StructField("city", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("pressure", IntegerType()),
    StructField("wind_speed", DoubleType()),
    StructField("cloud", IntegerType()),
    StructField("weather", StringType()),
    StructField("timestamp", StringType()),
    StructField("month", IntegerType()),
    StructField("hour", IntegerType())
])

# Read from Kafka (Using v2 topic if you created one, otherwise just "weather_stream")
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "weather_stream_v2") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parsing JSON and casting Time
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), weather_schema).alias("data")) \
    .select("data.*") \
    .withColumn("timestamp", to_timestamp("timestamp"))

# Cleaning features for model injection
ml_features_df = parsed_df \
    .withColumnRenamed("temperature", "temperature_celsius") \
    .withColumn("humidity", col("humidity").cast("double")) \
    .withColumn("pressure", col("pressure").cast("double")) \
    .withColumn("wind_speed", col("wind_speed").cast("double")) \
    .withColumn("cloud", col("cloud").cast("double")) \
    .dropna(subset=["latitude", "longitude", "month", "hour", "temperature_celsius", "humidity", "pressure", "wind_speed", "cloud"])

# Applying Model virt sensor
prediction_df = model.transform(ml_features_df)

# THE FIX: Clamp negative PM2.5 predictions to 0.0 using PySpark's 'when' function
prediction_df = prediction_df.withColumn(
    "predicted_pm2_5", 
    when(col("prediction") < 0, 0.0).otherwise(col("prediction"))
)

# Structuring the final output
final_output = prediction_df.select(
    "city",
    "latitude",
    "longitude",
    col("temperature_celsius").alias("temperature"),
    "humidity",
    "pressure",
    "wind_speed",
    "cloud",
    "predicted_pm2_5",
    "timestamp"
)

# THE FACELIFT: Streaming Anomaly Detection via Windowed Variance
windowed_analytics = final_output \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        "city"
    ).agg(
        first("latitude").alias("lat"),
        first("longitude").alias("lon"),
        avg("temperature").alias("avg_temp"),
        stddev("temperature").alias("temp_volatility"), # Track how wildly temp swings
        avg("predicted_pm2_5").alias("avg_pm2_5"),
        count("*").alias("events_processed")
    ).withColumn(
        "is_anomaly",
        when(col("temp_volatility") > 1.0, True).otherwise(False) # Flag sudden micro-climate spikes
    ).fillna(0.0, subset=["temp_volatility"]) # Handle nulls if only 1 event in window

# Real-time analytics to Console
console_query = windowed_analytics.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", False) \
    .trigger(processingTime="15 seconds") \
    .start()

# Raw Parquet to Hadoop 3.4.3 HDFS Data Lake (Updated path to v2 to avoid checkpoint crashes)
hdfs_query = final_output.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "hdfs://localhost:9000/weather-data/live_lake_v2") \
    .option("checkpointLocation", "hdfs://localhost:9000/weather-data/checkpoints/live_lake_v2") \
    .trigger(processingTime="10 seconds") \
    .start()

spark.streams.awaitAnyTermination()