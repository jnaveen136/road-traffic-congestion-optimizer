import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Simulate fake traffic data
np.random.seed(42)
data = pd.DataFrame({
    'time_of_day': np.random.randint(0, 24, 100),
    'weather': np.random.choice(['Clear', 'Rain', 'Fog'], 100),
    'event': np.random.choice([0, 1], 100),
    'road_id': np.random.choice(['A-B', 'B-C', 'A-C'], 100),
    'congestion_factor': np.random.uniform(1.0, 2.5, 100)
})

# Encode text columns into numbers (0s and 1s)
data_encoded = pd.get_dummies(data, columns=['weather', 'road_id'])

# Separate features (X) and target (y)
X = data_encoded.drop('congestion_factor', axis=1)
y = data_encoded['congestion_factor']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Build road network
G = nx.DiGraph()
G.add_edge("A", "B", base_time=4)
G.add_edge("B", "C", base_time=2)
G.add_edge("A", "C", base_time=6)

# Function to predict congestion for each road
def predict_congestion(time_of_day, weather, event, road_id):
    input_df = pd.DataFrame([{
        'time_of_day': time_of_day,
        'event': event,
        'weather_' + weather: 1,
        'road_id_' + road_id: 1
    }])
    for col in X.columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[X.columns]
    return model.predict(input_df)[0]

# 🧑‍💻 Get user input
print("🚦 Road Traffic Optimizer")
time_of_day = int(input("Enter hour of day (0 to 23): "))
weather = input("Enter weather (Clear, Rain, Fog): ").capitalize()
event_input = input("Is there a nearby event? (yes/no): ").lower()
event = 1 if event_input == "yes" else 0
source = input("Enter source (A/B/C): ").upper()
target = input("Enter destination (A/B/C): ").upper()

# Update weights based on user input
for u, v in G.edges():
    road_id = f"{u}-{v}"
    factor = predict_congestion(time_of_day, weather, event, road_id)
    G[u][v]['weight'] = G[u][v]['base_time'] * factor

# Find and print shortest path
try:
    path = nx.dijkstra_path(G, source=source, target=target, weight='weight')
    total_time = nx.dijkstra_path_length(G, source=source, target=target, weight='weight')
    print("✅ Best Route:", " → ".join(path))
    print("⏱️ Estimated Travel Time:", round(total_time, 2), "minutes")
except nx.NetworkXNoPath:
    print("❌ No route found between", source, "and", target)

