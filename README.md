# Health Care Disease Prediction Web Application

This is a Flask-based web application that predicts various diseases using machine learning models. The project includes modules for:

- **Disease Predictions and Medications**
- **Brain Tumor Detection (KYH)**
- **Heart Disease Prediction**
- **Liver Disease Prediction**
- Some features are still in the development phase.

## Features

- Predict diseases based on user input symptoms or medical parameters.
- Upload images for brain tumor detection.
- Interactive web UI built using Flask and HTML templates.
- Models are pre-trained using datasets and loaded at runtime.
- Provides disease description, precautions, medications, diet, and workout suggestions.

## Project Structure

```
.
├── app.py                     # Flask main application
├── Disease_Prediction.py     # Logic for symptom-based disease prediction
├── heart.py                  # Heart disease ML model and prediction
├── liver.py                  # Liver disease ML model and prediction
├── templates/                # HTML templates
├── static/                   # Static files (images, CSS)
└── requirements.txt          # Python dependencies
```

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open your browser and visit
```
http://127.0.0.1:5000/
```

## Routes Overview

- `/` – Home page
- `/kyd` – Kidney disease input page
- `/predict` – Process symptoms for kidney disease
- `/kyh_diseases` – Brain tumor section
- `/upload` – Upload brain MRI image
- `/heart` – Heart disease prediction form
- `/heart_predict` – Heart prediction logic
- `/liver` – Liver disease prediction form
- `/liver_predict` – Liver prediction logic
- `/request_appointment` – Appointment page
- `/about` – About page
- `/contact` – Contact page

## Technologies Used

- **Python 3.x**
- **Flask**
- **TensorFlow / Keras** 
- **scikit-learn** 
- **HTML/CSS**
