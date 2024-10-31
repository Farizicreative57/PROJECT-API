import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

path = r'./python-service/Model'
train_dir = os.path.join(path, 'Training')
val_dir = os.path.join(path, 'Validation')

datagen = ImageDataGenerator(
    rescale=16./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
)

test_datagen = ImageDataGenerator(
    rescale=16./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(60, 80),
    batch_size=32,
    class_mode='categorical',
)
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(60, 80),
    batch_size=32,
    class_mode='categorical',
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(60, 80, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=30,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=2)

model.save("python-service/model_path.h5")