import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

lily_path = './flower_images/Lilly'
lotus_path = './flower_images/Lotus'
orchid_path = './flower_images/Orchid'
sunflower_path = './flower_images/Sunflower'
tulip_path = './flower_images/Tulip'


# Fazer filtros na mao
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    
                            )

train_generator = datagen.flow_from_directory(
    './flower_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


# Aumentar a rede
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    # Qual a mehor ativacao?
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


# Quais metricas usar?
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Quantas epocas treinar?
model.fit(train_generator, epochs=10)

model.save('flower_classifier_model')

loaded_model = tf.keras.models.load_model('flower_classifier_model')

def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    prediction = loaded_model.predict(img_array)
    class_names = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_class = class_names[predicted_class_index]
    predicted_percentage = prediction[0][predicted_class_index] * 100

    return predicted_class, predicted_percentage

image_path = './flower_images/Lotus/0cd4d2960b.jpg'
predicted_class, predicted_percentage = classify_image(image_path)
print(f"The predicted class for {image_path} is: {predicted_class}")
print(f"The percentage of the predicted class is: {predicted_percentage:.2f}%")

