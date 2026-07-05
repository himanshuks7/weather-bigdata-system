from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("WeatherML_PM25_Training") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Initializing Spark 4.1.1 ML Pipeline...")

# Loading historical global dataset
df = spark.read.csv(
    "../data/GlobalWeatherRepository.csv",
    header=True,
    inferSchema=True
)

sanitized_columns = [c.replace('.', '_').replace('-', '_') for c in df.columns]
df = df.toDF(*sanitized_columns)

df = df.select(
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud",
    "air_quality_PM2_5"
).dropna()

df = df.withColumnRenamed("air_quality_PM2_5", "label") \
       .withColumnRenamed("pressure_mb", "pressure") \
       .withColumnRenamed("wind_kph", "wind_speed")

assembler = VectorAssembler(
    inputCols=[
        "temperature_celsius",
        "humidity",
        "pressure",
        "wind_speed",
        "cloud"
    ],
    outputCol="features"
)

# Distributed Random Forest implementation
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=75,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[assembler, rf])

print("Training Distributed Random Forest Model...")
model = pipeline.fit(df)

# Save to local or HDFS
model_path = "../ml/pm25_rf_model"
model.write().overwrite().save(model_path)

print(f"PM2.5 Prediction Model trained and saved successfully at: {model_path}")
spark.stop()