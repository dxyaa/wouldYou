import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("./datasets/estonia-passenger-list.csv")

# Features and target
features = ["Country", "Sex", "Age", "Category"]
target = "Survived"

# Fill missing values
data.fillna({"Country": "Unknown", "Age": data["Age"].median(), "Category": "Unknown"}, inplace=True)

# Preprocessing: One-Hot Encode categorical features
preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Apply preprocessing
X_encoded = preprocessor.fit_transform(data[["Country", "Sex", "Category"]])
X = np.hstack((X_encoded, data[["Age"]].values))  # Combine with Age
y = data[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- MP Neuron Model ----
class MPNeuron:
    def __init__(self, threshold=2):  # Default threshold
        self.threshold = threshold

    def predict(self, X):
        """Predicts based on threshold logic."""
        return np.where(np.sum(X, axis=1) >= self.threshold, 1, 0)

    def predict_single(self, input_data):
        """Predicts for a single passenger."""
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for prediction
        return self.predict(input_array)[0]

# Train MP Neuron (No learning, just setting a threshold)
mp_neuron = MPNeuron(threshold=2)

# ---- SIMPLIFIED PREDICTION FUNCTION ----
def predict_new_passenger(passenger_details):
    """Takes raw passenger details and predicts survival automatically."""
    df = pd.DataFrame([passenger_details], columns=features)

    # Encode input using the same preprocessor
    df_encoded = preprocessor.transform(df[["Country", "Sex", "Category"]])
    X_input = np.hstack((df_encoded, df[["Age"]].values))  # Combine with Age

    # Predict using MP Neuron
    prediction = mp_neuron.predict_single(X_input)
    return "Survived" if prediction == 1 else "Not Survived"

# ---- TEST A NEW PASSENGER ----
new_passenger = ["Sweden", "M", 70, "P"]  # Raw values, no need to encode manually
print(predict_new_passenger(new_passenger))
