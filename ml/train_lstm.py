import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Initializing PyTorch Spatio-Temporal LSTM Pipeline...")

# 1. Load Data
df = pd.read_csv("../data/GlobalWeatherRepository.csv")

# Sanitize columns
df.columns = [c.replace('.', '_').replace('-', '_') for c in df.columns]

# Parse time and sort to ensure strict chronological sequences
df['parsed_time'] = pd.to_datetime(df['last_updated'])
df['month'] = df['parsed_time'].dt.month
df['hour'] = df['parsed_time'].dt.hour
df = df.sort_values(by=['location_name', 'parsed_time'])

# Select features
feature_cols = [
    "latitude", "longitude", "month", "hour", 
    "temperature_celsius", "humidity", "pressure_mb", 
    "wind_kph", "cloud"
]
target_col = "air_quality_PM2_5"

# Drop nulls to keep sequences clean
df = df.dropna(subset=feature_cols + [target_col])

# 2. Scale the Features (CRITICAL FOR NEURAL NETWORKS)
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
y = df[target_col].values

# Save the scaler so the Spark Stream can use the EXACT same mathematical transformation
os.makedirs("../ml", exist_ok=True)
joblib.dump(scaler, "../ml/lstm_scaler.pkl")

# 3. Create Sequences (Window Size = 10 events)
# We group by city to ensure we don't mix Tokyo's weather with London's in a single sequence
print("Building Spatio-Temporal Sequences...")
SEQ_LENGTH = 10
X_seq, y_seq = [], []

# Group by location to build isolated sequences
for _, group_df in df.groupby("location_name"):
    indices = group_df.index.tolist()
    if len(indices) < SEQ_LENGTH:
        continue
        
    for i in range(len(indices) - SEQ_LENGTH):
        # Grab the scaled features for this sequence window
        window_idx = indices[i : i + SEQ_LENGTH]
        X_seq.append(X_scaled[df.index.get_indexer(window_idx)])
        # The target is the PM2.5 value right after the sequence ends
        y_seq.append(y[df.index.get_loc(indices[i + SEQ_LENGTH])])

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).view(-1, 1)

print(f"Total sequences generated: {X_seq.shape[0]}")

# 4. Define the PyTorch LSTM Architecture
class WeatherDriftLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(WeatherDriftLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layers to decode the LSTM memory state into a PM2.5 prediction
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # out shape: (batch, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        # We only care about the memory state at the VERY LAST time step of the sequence
        last_time_step = lstm_out[:, -1, :] 
        
        x = self.relu(self.fc1(last_time_step))
        prediction = self.fc2(x)
        return prediction

# Initialize Model
INPUT_SIZE = len(feature_cols) # 9 features
HIDDEN_SIZE = 64
NUM_LAYERS = 2

model = WeatherDriftLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
EPOCHS = 15
BATCH_SIZE = 256
dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Training LSTM Sequence Model...")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.4f}")

# 6. Export for Spark Integration
# We use TorchScript (jit) to trace the model. This allows Spark to load and run 
# the model natively without needing to import the WeatherDriftLSTM class definition.
print("Tracing and exporting model for Spark Stateful Streaming...")
model.eval()
example_input = torch.rand(1, SEQ_LENGTH, INPUT_SIZE)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("../ml/pm25_lstm_traced.pt")

print("✅ Success: Scaler saved to '../ml/lstm_scaler.pkl' and Model saved to '../ml/pm25_lstm_traced.pt'")