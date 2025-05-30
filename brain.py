
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained model
saved_model = load_model("W:/Capstone/health_care/datasets/VGG_model.h5")

def check(input_img):
    print("Your image is: " + input_img)
    
    # Load and preprocess the image
    img = image.load_img("images/" + input_img, target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    # Predict the result
    output = saved_model.predict(img)
    status = output[0][0] == 1

    print("Prediction status:", status)
    return status
