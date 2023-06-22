import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import classification_report, f1_score , confusion_matrix
import visualkeras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization


model_name = 'flower_classifier_model_dag'
train_path = './a_train_images'
test_path = './a_test_images'


train_datagen = ImageDataGenerator(
    rescale=1./255,
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
    SeparableConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    SeparableConv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    SeparableConv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)


history = model.fit(train_generator,
                    epochs=50,
                    validation_data=test_generator,
                    verbose=1,
                    callbacks=[early_stopping, reduce_lr])

model.save(model_name)

# Define needed variables
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.savefig(f'./{model_name}_history.png')

print(model.summary())

# Visualize model
visualkeras.layered_view(model, legend=True, to_file=f'./{model_name}_model.png')

