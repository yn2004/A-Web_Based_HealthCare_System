import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to train the liver disease model
def train_model():
    # Load dataset
    df = pd.read_csv("W:/Capstone/health_care/datasets/indian_liver_patient.csv")
    
    # Fill missing values
    df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(df['Albumin_and_Globulin_Ratio'].mean())
    
    # Encode Gender (Male = 1, Female = 0)
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    # Define features and target variable
    X = df.drop(['Dataset'], axis=1)
    y = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)  # 1: liver disease, 0: no liver disease
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Print model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, X.columns

# Prediction function
def predict_liver_disease(input_data, model, feature_columns):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    prediction = model.predict(input_df)
    return "Liver disease" if prediction[0] == 1 else "No liver disease"
