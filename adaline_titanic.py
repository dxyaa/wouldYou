import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=1000):
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
        return np.where(linear_output >=0.5, 1, 0)

train_data = pd.read_csv("./datasets/titanic_train.csv")
test_data = pd.read_csv("./datasets/titanic_test.csv")

# Selecting relevant columns (removing "Name" and "Ticket")
columns = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
train_data = train_data[columns + ["Survived"]]
test_data = test_data[columns]

# Handle missing values
train_data.fillna({"Age": train_data["Age"].median(), "Fare": train_data["Fare"].median(), "Cabin": "Unknown", "Embarked": "S"}, inplace=True)
test_data.fillna({"Age": test_data["Age"].median(), "Fare": test_data["Fare"].median(), "Cabin": "Unknown", "Embarked": "S"}, inplace=True)

# Simplify Cabin feature
train_data["Cabin"] = np.where(train_data["Cabin"] == "Unknown", 0, 1)
test_data["Cabin"] = np.where(test_data["Cabin"] == "Unknown", 0, 1)

# Define categorical and numerical features
categorical_features = ["Sex", "Embarked"]
numerical_features = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Cabin"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Apply transformations
X_train = preprocessor.fit_transform(train_data.drop(columns=["Survived"]))
y_train = train_data["Survived"].values
X_test = preprocessor.transform(test_data)

# Train Adaline model
adaline = Adaline(learning_rate=0.001, epochs=1500)
adaline.fit(X_train, y_train)

# Predictions
y_pred = adaline.predict(X_test)

# Function to predict a new passenger
def predict_new_passenger(passenger_details):
    df = pd.DataFrame([passenger_details], columns=columns)
    
    # Process new passenger data
    df["Cabin"] = np.where(df["Cabin"] == "Unknown", 0, 1)
    
    df = preprocessor.transform(df)
    return "Survived" if adaline.predict(df)[0] == 1 else "Not Survived"

# Example usage
new_passenger = [1001, 1, "male",30, 0, 0, 25, "Unknown", "S"]
print(predict_new_passenger(new_passenger))
