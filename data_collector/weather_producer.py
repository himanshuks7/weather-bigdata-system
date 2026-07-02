import json
import time
import random
import requests
from kafka import KafkaProducer
from datetime import datetime, timezone

API_KEY = "YOUR_OPENWEATHER_API_KEY"
KAFKA_TOPIC = "weather_stream_v3"
KAFKA_BROKER = "localhost:9092"     

CITIES = [
    "Delhi",
    "Mumbai",
    "Chennai",
    "Bangalore",
    "Kolkata",
    "Hyderabad",
    "London",
    "New York",
    "Tokyo"
]

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)