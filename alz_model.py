import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2

# Directory paths
dataset_dir = '/Users/yunusajib/Downloads/Alzheimer_s Dataset'  # Update this to your dataset path
train_dir = f'{dataset_dir}/train'
test_dir = f'{dataset_dir}/test'  # Assuming you want to use the Test set as validation

# Image parameters
img_width, img_height = 150, 150  # Adjust based on your dataset
batch_size = 32

# Data preprocessing with a validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # 20% of the training data will be used for validation

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')  # Set as training data

validation_generator = datagen.flow_from_directory(
    train_dir,  # Same directory as training data, but using the validation split
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')  # Set as validation data

# Test data generator without validation split
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)  # Typically, you don't need to shuffle the test set

# CNN Model Architecture
l2_reg_factor = 0.001
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3), kernel_regularizer=l2(l2_reg_factor)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg_factor)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg_factor)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(l2_reg_factor)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Train the model with validation split
epochs = 50  # Adjust as needed
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

model_save_path = '/Users/yunusajib/Downloads/Alzheimer_s Dataset' 
model.save(model_save_path)
print(f'Model saved to {model_save_path}')


from keras.models import load_model

# Load the saved model
loaded_model = load_model(model_save_path)
print('Model loaded successfully')

# Evaluate the loaded model with the test set
test_loss, test_acc = loaded_model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

import numpy as np
import matplotlib.pyplot as plt

# Load the model (if not already loaded)
# model = load_model(model_save_path)

# Get a batch of images from the test set
test_images, test_labels = next(test_generator)

# Make predictions
predictions = model.predict(test_images)
predictions = predictions.flatten()
predicted_classes = predictions > 0.5  # Assuming binary classification with sigmoid activation

# Mapping of indices to class labels (based on the order in the test_generator's class_indices)
class_labels = list(test_generator.class_indices.keys())

# Display a sample of images, their true labels, and the model's predictions
plt.figure(figsize=(10, 10))
for i in range(9):  # Display the first 9 images and predictions
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"True: {class_labels[int(test_labels[i])]}, Pred: {class_labels[int(predicted_classes[i])]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
