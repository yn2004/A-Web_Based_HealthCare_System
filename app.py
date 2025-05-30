import os
from flask import Flask, request, render_template, url_for
import numpy as np
import pickle
from Disease_Prediction import get_predicted_value, helper
from heart import train_model, predict_heart_disease
# from brain import check
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image

from tensorflow.keras.models import load_model 

from liver import train_model, predict_liver_disease

# from lung import train_lung_cancer_model

# from lung import train_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

########################################### KYD #################################
@app.route("/kyd")
def kyd():
    return render_template("kyd.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('kyd.html', message=message)
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            my_precautions = [i for i in precautions[0]]

            return render_template('kyd.html', predicted_disease=predicted_disease, 
                                   dis_des=dis_des, my_precautions=my_precautions, 
                                   medications=medications, my_diet=rec_diet, 
                                   workout=workout)
    
    return render_template('kyd.html')

########################################### KYH #################################
@app.route('/kyh_diseases')
def kyh_diseases():
    return render_template("KYH/kyh_diseases.html")

#############################  BRAIN ##################

# Load the pre-trained model
# saved_model = load_model("W:/Capstone/health_care/datasets/VGG_model.h5")
saved_model = load_model("W:/Capstone/health_care/datasets/VGG_model.h5", compile=False)


def check(input_img):
    # Load and preprocess the image
    img_path = os.path.join('static', 'images', input_img)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    # Predict the result
    output = saved_model.predict(img)
    status = output[0][0] == 1

    return status

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Ensure the directory exists
            upload_folder = os.path.join('static', 'images')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            # Save the file to the upload folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Check the tumor status
            predvalue = check(filename)

            # Generate URL for the uploaded image
            image_url = url_for('static', filename='images/' + filename)

            return render_template("KYH/brain.html", image_name=image_url, predvalue=predvalue)

    return render_template("KYH/brain.html", image_name=None, predvalue=None)

#############################  HEART ##################
model = train_model()

@app.route("/heart")
def heart():
    return render_template("KYH/heart.html")

@app.route('/heart_predict', methods=['GET', 'POST'])
def heart_predict():
    if request.method == 'POST':
        try:
            male = int(request.form['male'])
            age = int(request.form['age'])
            education = int(request.form['education'])
            currentSmoker = int(request.form['currentSmoker'])
            cigsPerDay = int(request.form['cigsPerDay'])
            BPMeds = int(request.form['BPMeds'])
            prevalentStroke = int(request.form['prevalentStroke'])
            prevalentHyp = int(request.form['prevalentHyp'])
            diabetes = int(request.form['diabetes'])
            totChol = int(request.form['totChol'])
            sysBP = float(request.form['sysBP'])
            BMI = float(request.form['BMI'])
            heartRate = int(request.form['heartRate'])
            glucose = int(request.form['glucose'])

            input_data = [male, age, education, currentSmoker, cigsPerDay, BPMeds,
                          prevalentStroke, prevalentHyp, diabetes, totChol, 
                          sysBP, BMI, heartRate, glucose]

            prediction = predict_heart_disease(model, input_data)

            if prediction == 1:
                result_message = "The person is at risk of heart disease."
            else:
                result_message = "The person is not at risk of heart disease."

            return render_template('KYH/heart.html', result=result_message)
        
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('KYH/heart.html', result=error_message)

    return render_template('KYH/heart.html')

#############################  LIVER ##################

model, feature_columns = train_model()

@app.route('/liver', methods=['GET', 'POST'])
def liver_predict():
    if request.method == 'POST':
        try:
            # Collect form data
            age = int(request.form['age'])
            gender = int(request.form['gender'])  # Male = 1, Female = 0
            total_bilirubin = float(request.form['total_bilirubin'])
            direct_bilirubin = float(request.form['direct_bilirubin'])
            alkphos = int(request.form['alkphos'])
            sgpt = int(request.form['sgpt'])
            sgot = int(request.form['sgot'])
            total_proteins = float(request.form['total_proteins'])
            albumin = float(request.form['albumin'])
            albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])
            
            # Prepare input data for the model
            input_data = [age, gender, total_bilirubin, direct_bilirubin, alkphos, sgpt, sgot, total_proteins, albumin, albumin_and_globulin_ratio]
            
            # Make prediction using the trained model
            prediction = predict_liver_disease(input_data, model, feature_columns)
            
            # Render the result to the HTML template
            return render_template('KYH/liver.html', result=prediction)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('KYH/liver.html', result=error_message)
    
    return render_template('KYH/liver.html')

#############################  LUNG ##################

# model = train_model()

# @app.route('/lung', methods=['POST'])
# def lung_predict():
#     try:
#         age = int(request.form['age'])
#         smoking = int(request.form['smoking'])
#         yellow_fingers = int(request.form['yellow_fingers'])
#         anxiety = int(request.form['anxiety'])
#         peer_pressure = int(request.form['peer_pressure'])
#         chronic_disease = int(request.form['chronic_disease'])
#         fatigue = int(request.form['fatigue'])
#         allergy = int(request.form['allergy'])
#         wheezing = int(request.form['wheezing'])
#         alcohol = int(request.form['alcohol'])
#         coughing = int(request.form['coughing'])
#         shortness_of_breath = int(request.form['shortness_of_breath'])
#         swallowing_difficulty = int(request.form['swallowing_difficulty'])
#         chest_pain = int(request.form['chest_pain'])

#         # Organize input features into an array
#         features = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure,
#                               chronic_disease, fatigue, allergy, wheezing, alcohol,
#                               coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

#         # Make prediction
#         prediction = model.predict(features)
#         result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer Detected"

#         return render_template('KYH/lung.html', result=result)

#     except Exception as e:
#         return render_template('KYH/lung.html', result=f"Error: {str(e)}")


######################################### REQ_AP ################################
@app.route('/request_appointment')
def request_appointment():
    return render_template("request_appointment.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.run(debug=True)
