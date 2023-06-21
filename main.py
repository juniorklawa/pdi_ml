import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



test_path = './raw_test_images'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

loaded_model = tf.keras.models.load_model('flower_classifier_model_raw')


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


print(
    f"The accuracy of the model is: {loaded_model.evaluate(test_generator)[1]}")
