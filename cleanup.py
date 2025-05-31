import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import pandas as pd


cnn = tf.keras.models.load_model('models/trained_plant_disease_model.keras')

image_path = './temp.jpg'
# Reading an image in default mode
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB
# Displaying the image 
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()


image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)

print(predictions)


result_index = np.argmax(predictions) #Return index of max element
print(result_index)

print(cnn.summary())

df = pd.read_csv('Labels/labels.csv')
plant_disease_class_name = list(df['Plant_Disease_Labels'])


model_prediction = plant_disease_class_name[result_index]
print(model_prediction)
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()


### Pest detection

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image


df = pd.read_csv('Labels/pest_labels.csv')
pest_class_name = list(df['Pest_labels'])


# Load the model (No need for custom objects!)
model = load_model("models/pest_prediction_model.joblib")  # Update with your path

print("✅ Model loaded successfully!")


# ✅ Step 2: Define function to preprocess input image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load image
    img_array = img_to_array(img) / 255.0  # Convert to array & normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_pest(image_path):
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)  # Get prediction probabilities
    predicted_class = np.argmax(preds, axis=1)[0]  # Get highest prob index
    
    # Get class name
    predicted_label = index_to_class.get(predicted_class, "Unknown")

    print(f"🎯 Predicted Class: {predicted_label} (Class Index: {predicted_class})")
    print("🔢 Prediction Probabilities:", preds)

# ✅ Step 5: Run Prediction
image_path = "Pest_Dataset/Ampelophaga/43864.jpg"  # Change to your test image path
predict_pest(image_path)