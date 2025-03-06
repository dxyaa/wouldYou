import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load dataset
data = pd.read_csv("./datasets/estonia-passenger-list.csv")

# Features and target
features = ["Country", "Sex", "Age", "Category"]
target = "Survived"

# Fill missing values
data.fillna({"Country": "Unknown", "Age": data["Age"].median(), "Category": "Unknown"}, inplace=True)

# Normalize Age
scaler = MinMaxScaler()
data["Age"] = scaler.fit_transform(data[["Age"]])

# Encode Sex (Make Male = 1, Female = 2 so it's weighted higher)
data["Sex"] = data["Sex"].map({"M": 1, "F": 2})  # Assign more weight to Female

# One-Hot Encode categorical features
preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = preprocessor.fit_transform(data[["Country", "Category"]])  # Exclude Sex

# Normalize one-hot encoding
X_encoded = X_encoded / X_encoded.shape[1]

# Multiply Sex and Age by weights
X_weighted = np.hstack((X_encoded, (data[["Age"]] * 2).values, (data[["Sex"]] * 3).values))  # Age × 2, Sex × 3

y = data[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)

# ---- MP Neuron with Weighted Features ----
class MPNeuron:
    def __init__(self):
        self.threshold = None

    def find_best_threshold(self, X, y):
        best_threshold = 0
        best_accuracy = 0

        for t in np.linspace(0, X.max(), 50):  # Iterate through possible thresholds
            y_pred = np.where(np.sum(X, axis=1) >= t, 1, 0)
            accuracy = np.mean(y_pred == y)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        self.threshold = best_threshold
        print(f"Optimal Threshold Found: {self.threshold}")

    def predict(self, X):
        return np.where(np.sum(X, axis=1) >= self.threshold, 1, 0)

    def predict_single(self, input_data):
        input_array = np.array(input_data).reshape(1, -1)
        return self.predict(input_array)[0]

# Train MP Neuron
mp_neuron = MPNeuron()
mp_neuron.find_best_threshold(X_train, y_train)

# ---- Prediction Function ----
def predict_new_passenger(passenger_details):
    df = pd.DataFrame([passenger_details], columns=features)

    # Normalize Age
    df["Age"] = scaler.transform(df[["Age"]])

    # Encode Sex (F = 2, M = 1)
    df["Sex"] = df["Sex"].map({"M": 2, "F": 3   })

    # Encode categorical values
    df_encoded = preprocessor.transform(df[["Country", "Category"]])
    df_encoded = df_encoded / df_encoded.shape[1]

    # Apply feature weights
    X_input = np.hstack((df_encoded, (df[["Age"]] * 3).values, (df[["Sex"]] * 2).values))

    # Predict
    prediction = mp_neuron.predict_single(X_input)
    return "Survived" if prediction == 1 else "Not Survived"

# ---- TEST A NEW PASSENGER ----
new_passenger = ["Estonia", "M", 10, "C"]  # A female passenger, Age 30
print(predict_new_passenger(new_passenger))
