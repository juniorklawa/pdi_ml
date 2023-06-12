import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os


# Split the dataset into 70% train set and 30% test set
def split_data(path):
    all_images = os.listdir(path)
    train_images, test_images = train_test_split(all_images, test_size=0.3, random_state=42)
    return train_images, test_images


lily_train, lily_test = split_data('./flower_images/Lilly')
lotus_train, lotus_test = split_data('./flower_images/Lotus')
orchid_train, orchid_test = split_data('./flower_images/Orchid')
sunflower_train, sunflower_test = split_data('./flower_images/Sunflower')
tulip_train, tulip_test = split_data('./flower_images/Tulip')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create train and test generator using the split image file names
train_generator = train_datagen.flow_from_directory(
    './flower_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    './flower_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_generator, epochs=20, validation_data=test_generator, callbacks=[early_stopping])

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