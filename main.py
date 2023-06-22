import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns



test_path = './a_test_images'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

loaded_model = tf.keras.models.load_model('flower_classifier_model_dag')


def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    prediction = loaded_model.predict(img_array)
    class_names = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_class = class_names[predicted_class_index]

    return predicted_class, predicted_class_index, prediction




# print(f"The accuracy of the model is: {loaded_model.evaluate(test_generator)[1]}")


# PREDICTING A SINGLE IMAGE
image_path = './extra_images/tulipa_1.jpeg'
predicted_class, predicted_class_index, prediction = classify_image(image_path)
print(f"The predicted class is: {predicted_class} with a confidence of {prediction[0][predicted_class_index]}")



