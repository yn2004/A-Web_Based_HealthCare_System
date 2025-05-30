import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Function to train the model
def train_model():
    # Load your dataset
    df = pd.read_csv(r'W:\Capstone\health_care\datasets\framingham.csv')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Define features and target variable
    features = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 
                'sysBP', 'BMI', 'heartRate', 'glucose']
    target = 'TenYearCHD'
    
    X = df[features]
    y = df[target]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Function to predict heart disease
def predict_heart_disease(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
                                                   'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
                                                   'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose'])
    
    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]
