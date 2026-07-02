import json
import time
import random
import requests
from kafka import KafkaProducer
from datetime import datetime, timezone

API_KEY = "YOUR_OPENWEATHER_API_KEY"
KAFKA_TOPIC = "weather_stream_v3"
KAFKA_BROKER = "localhost:9092"