import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil

# Create separate directories for training and testing images
def create_train_test_dirs(base_path, train_path, test_path):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    flower_folders = os.listdir(base_path)

    for folder in flower_folders:
        train_flower_path = os.path.join(train_path, folder)
        test_flower_path = os.path.join(test_path, folder)
        
        os.makedirs(train_flower_path, exist_ok=True)
        os.makedirs(test_flower_path, exist_ok=True)

        flower_images = os.listdir(os.path.join(base_path, folder))
        train_images, test_images = train_test_split(flower_images, test_size=0.3, random_state=42)

        for train_image in train_images:
            src = os.path.join(base_path, folder, train_image)
            dst = os.path.join(train_flower_path, train_image)
            shutil.copy(src, dst)

        for test_image in test_images:
            src = os.path.join(base_path, folder, test_image)
            dst = os.path.join(test_flower_path, test_image)
            shutil.copy(src, dst)

base_path = './flower_images'
train_path = './train_images'
test_path = './test_images'

# Create training and testing directories
create_train_test_dirs(base_path, train_path, test_path)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
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

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stopping])

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

image_path = './test_images/Lotus/some_lotus_image.jpg'  # Replace 'some_lotus_image.jpg' with the correct file name
predicted_class, predicted_percentage = classify_image(image_path)
print(f"The predicted class for {image_path} is: {predicted_class}")
print(f"The percentage of the predicted class is: {predicted_percentage:.2f}%")