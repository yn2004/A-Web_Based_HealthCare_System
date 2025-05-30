import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('W:\Capstone\health_care\datasets\survey lung cancer.csv')

# Define features (X) and target (y)
X = data[['Age', 'Smoking', 'Yellow_Fingers', 'Anxiety', 'Peer_Pressure',
          'Chronic_Disease', 'Fatigue', 'Allergy', 'Wheezing', 'Alcohol',
          'Coughing', 'Shortness_of_Breath', 'Swallowing_Difficulty', 'Chest_Pain']]
y = data['Lung_Cancer']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
