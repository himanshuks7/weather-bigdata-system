import json
import time
import random
import requests
from kafka import KafkaProducer
from datetime import datetime, timezone

API_KEY = "MYAPI"
KAFKA_TOPIC = "weather_stream"
KAFKA_BROKER = "localhost:9092" 

cities = [
    "Delhi", "Mumbai", "Chennai", "Bangalore", "Kolkata", 
    "Hyderabad", "London", "New York", "Tokyo"
]

# Initializing Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks="all",
    retries=3
)

def fetch_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        return {
            "city": city,
            "temperature": float(data["main"]["temp"]),
            "humidity": float(data["main"]["humidity"]),
            "pressure": int(data["main"]["pressure"]),
            "wind_speed": float(data["wind"]["speed"]),
            "cloud": int(data["clouds"]["all"]), 
            "weather": str(data["weather"][0]["main"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API Error for {city}: {e}")
        return None

print(f"Starting Live Data Ingestion to Kafka Topic: '{KAFKA_TOPIC}'...")

try:
    while True:
        for city in cities:
            base_data = fetch_weather(city)

            if base_data:
                # Simulating high-velocity streaming data (20 events per city)
                for i in range(20):
                    record = base_data.copy()
                    
                    # Injecting micro-variance for dynamic real-time visualization
                    record["temperature"] = round(record["temperature"] + random.uniform(-0.8, 0.8), 2)
                    record["humidity"] = round(record["humidity"] + random.uniform(-1.5, 1.5), 2)
                    record["wind_speed"] = round(record["wind_speed"] + random.uniform(-0.5, 0.5), 2)

                    producer.send(KAFKA_TOPIC, record)
                    print(f"Sent: {record['city']} | Temp: {record['temperature']}°C | Wind: {record['wind_speed']} kph")

        producer.flush()
        print("Batch complete.\n")
        time.sleep(30)
        
except KeyboardInterrupt:
    print("\nShutting down Kafka Producer...")
    producer.close()