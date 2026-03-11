import tensorflow as tf
from tensorflow.keras import layers, models
#import os

# 1. Setup Data Pipeline
data_dir = "dataset" 
img_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# --- NEW: DATA AUGMENTATION BLOCK ---
# This helps the model work in "messy, real-world fields" 
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])
# -------------------------------------

# 2. Build CNN Architecture [cite: 14]
model = models.Sequential([
    # Apply augmentation first
    data_augmentation, 
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train and Save
# Increasing epochs to 10 as augmentation requires more time to learn
model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save("plant_model.h5")
print("Model Saved Successfully!")
