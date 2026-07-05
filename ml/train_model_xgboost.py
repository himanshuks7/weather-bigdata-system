from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, month, hour
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

# Initialize Spark 4.1.1 Session
spark = SparkSession.builder \
    .appName("WeatherML_PM25_Training_SpatioTemporal") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Initializing Spark 4.1.1 ML Pipeline with Spatio-Temporal Encoding...")

# Load historical global dataset
df = spark.read.csv(
    "data/GlobalWeatherRepository.csv",
    header=True,
    inferSchema=True
)

# Sanitize columns (fixes '.' and '-' in names like 'air_quality_PM2.5')
sanitized_columns = [c.replace('.', '_').replace('-', '_') for c in df.columns]
df = df.toDF(*sanitized_columns)

# Feature Engineering: Extracting Temporal Features from 'last_updated'
df = df.withColumn("parsed_time", to_timestamp(col("last_updated"), "yyyy-MM-dd HH:mm"))
df = df.withColumn("month", month(col("parsed_time")))
df = df.withColumn("hour", hour(col("parsed_time")))

# Select core features + New Spatio-Temporal Features
df = df.select(
    "latitude",
    "longitude",
    "month",
    "hour",
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud",
    "air_quality_PM2_5"
).dropna()

# Standardize column names to perfectly match the Kafka streaming JSON schema
df = df.withColumnRenamed("air_quality_PM2_5", "label") \
       .withColumnRenamed("pressure_mb", "pressure") \
       .withColumnRenamed("wind_kph", "wind_speed")

# Assemble all features into a single ML vector
assembler = VectorAssembler(
    inputCols=[
        "latitude",
        "longitude",
        "month",
        "hour",
        "temperature_celsius",
        "humidity",
        "pressure",
        "wind_speed",
        "cloud"
    ],
    outputCol="features"
)

# THE UPGRADE: Gradient-Boosted Trees (XGBoost equivalent)
# This model learns sequentially, making it highly accurate for spatial drift
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=120,       # Number of boosting stages (trees)
    maxDepth=7,        # Keep depth moderate to prevent overfitting on specific coordinates
    stepSize=0.05,     # Learning rate - smaller step size with more iterations yields better accuracy
    seed=42
)

pipeline = Pipeline(stages=[assembler, gbt])

print("Training Gradient-Boosted Spatio-Temporal Model...")
model = pipeline.fit(df)

# Save to local or HDFS for the streaming engine
model_path = "ml/pm25_gbt_model"
model.write().overwrite().save(model_path)

print(f"Gradient-Boosted PM2.5 Prediction Model trained and saved successfully at: {model_path}")
spark.stop()