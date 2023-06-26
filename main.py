import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np



test_path = './a_test_images'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

models_names = [
    'flower_classifier_model_dag_resnet',
    'flower_classifier_model_raw_resnet',
    'flower_classifier_model_raw',
    'flower_classifier_model_dag'
]


def classify_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_names = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_class = class_names[predicted_class_index]

    return predicted_class, predicted_class_index, prediction


# generate graphic with each model accuracy in models_names
def generate_accuracy_graphic(models_names):
    # Definir cores para cada modelo
    colors = ['red', 'blue', 'green', 'orange']

    labels = ['ResNet DAG', 'ResNet RAW', 'Manual RAW', 'Manual DAG']

    accuracy = []
    for model_name in models_names:
        loaded_model = tf.keras.models.load_model(model_name)
        accuracy.append(loaded_model.evaluate(test_generator)[1])

    indices = np.arange(len(models_names))

    plt.bar(indices, accuracy, color=colors, tick_label=labels)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.title('Accuracy of each model')

    plt.savefig('accuracy.png')


def predict_image(image_path):
    predicted_classes = []
    labels = ['ResNet DAG', 'ResNet RAW', 'Manual RAW', 'Manual DAG']
    #  print(f'{labels[i]}: {predicted_class} - {prediction[0][predicted_class_index]*100}%')
    for i in range(4):
        print(f'Carregando modelo: {labels[i]}')
        loaded_model = tf.keras.models.load_model(models_names[i])
        predicted_class, predicted_class_index, prediction = classify_image(image_path, loaded_model)
        predicted_classes.append(predicted_class)
        print(f'{labels[i]}: {predicted_class} - {prediction[0][predicted_class_index]*100}%')






 

# generate_accuracy_graphic(models_names)
predict_image('./extra_images/lotus.jpeg')
        


   






