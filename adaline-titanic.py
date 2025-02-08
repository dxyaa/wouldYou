import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score

class Adaline:
    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            error = y - linear_output
            
            self.weights += self.learning_rate * np.dot(X.T, error) / n_samples
            self.bias += self.learning_rate * np.sum(error) / n_samples
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.5, 1, 0)

# Load pre-split datasets
train_data = pd.read_csv("./datasets/titanic_train.csv")
test_data = pd.read_csv("./datasets/titanic_test.csv")

# Selecting required columns
columns = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
train_data = train_data[columns + ["Survived"]]
test_data = test_data[columns]

# Handling missing values
train_data.fillna({"Age": train_data["Age"].median(), "Fare": train_data["Fare"].median(), "Cabin": "Unknown", "Embarked": "S"}, inplace=True)
test_data.fillna({"Age": test_data["Age"].median(), "Fare": test_data["Fare"].median(), "Cabin": "Unknown", "Embarked": "S"}, inplace=True)

# Encoding categorical data
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
categorical_cols = ["Sex", "Embarked", "Cabin", "Ticket", "Name"]
train_data[categorical_cols] = encoder.fit_transform(train_data[categorical_cols])
test_data[categorical_cols] = encoder.transform(test_data[categorical_cols])

# Splitting dataset
X_train = train_data.drop(columns=["Survived"]).values
y_train = train_data["Survived"].values
X_test = test_data.values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training Adaline model
adaline = Adaline()
adaline.fit(X_train, y_train)

y_pred = adaline.predict(X_test)

# Function to predict a new passenger
def predict_new_passenger(passenger_details):
    df = pd.DataFrame([passenger_details], columns=columns)
    df[categorical_cols] = encoder.transform(df[categorical_cols])
    df = scaler.transform(df.values)
    return "Survived" if adaline.predict(df)[0] == 1 else "Not Survived"

# Example usage
new_passenger = [1001, 1, "John Doe", "female", 5, 0, 0, "A/5 21171", 25, "Unknown", "S"]
print(predict_new_passenger(new_passenger))
