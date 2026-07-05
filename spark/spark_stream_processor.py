from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, window, avg, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml import PipelineModel

# Initialize Spark 4.1.1 Session for Structured Streaming
spark = SparkSession.builder \
    .appName("Weather_AI_Stream_Processor") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("⚡ Spark Structured Streaming Initialized.")

# Load the pre-trained Virtual Sensor model
model = PipelineModel.load("../ml/pm25_rf_model")

# Define exact schema for deterministic parsing
weather_schema = StructType([
    StructField("city", StringType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("pressure", IntegerType()),
    StructField("wind_speed", DoubleType()),
    StructField("cloud", IntegerType()),
    StructField("weather", StringType()),
    StructField("timestamp", StringType())
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "weather_stream") \
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
    .dropna(subset=["temperature_celsius", "humidity", "pressure", "wind_speed", "cloud"])

# Applying Model virt sensor
prediction_df = model.transform(ml_features_df)

# Structuring the final output
final_output = prediction_df.select(
    "city",
    col("temperature_celsius").alias("temperature"),
    "humidity",
    "pressure",
    "wind_speed",
    "cloud",
    col("prediction").alias("predicted_pm2_5"),
    "timestamp"
)


# Handling late-arriving data and calculates rolling 5 min avgs
windowed_analytics = final_output \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        "city"
    ).agg(
        avg("temperature").alias("avg_temp"),
        avg("predicted_pm2_5").alias("avg_pm2_5"),
        count("*").alias("events_processed")
    )

#real-time analytics to Console
console_query = windowed_analytics.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", False) \
    .trigger(processingTime="15 seconds") \
    .start()

# Raw Parquet to Hadoop 3.4.3 HDFS Data Lake
hdfs_query = final_output.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "hdfs://localhost:9000/weather-data/live_lake") \
    .option("checkpointLocation", "hdfs://localhost:9000/weather-data/checkpoints/live_lake") \
    .trigger(processingTime="10 seconds") \
    .start()

spark.streams.awaitAnyTermination()