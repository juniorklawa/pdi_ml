import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_path = './test_images'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

loaded_model = tf.keras.models.load_model('flower_classifier_model')


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

    return predicted_class


# Replace 'some_lotus_image.jpg' with the correct file name
image_path = './test_images/Lotus/0cad97f7dc.jpg'
predicted_class = classify_image(image_path)
print(f"The predicted class for {image_path} is: {predicted_class}")
# print the accuracy of the model using loaded_model
print(
    f"The accuracy of the model is: {loaded_model.evaluate(test_generator)[1]}")
