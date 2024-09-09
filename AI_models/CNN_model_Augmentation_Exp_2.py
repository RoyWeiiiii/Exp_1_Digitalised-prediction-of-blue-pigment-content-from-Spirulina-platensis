import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow_addons.metrics import RSquare
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data Preparation
main_data_dir = "D:\RonStuff\Experiment 2\CNN"
media_folders = ["AF6", "Bg-11", "Zarrouk"]

images = []
cpc_concentrations = []

desired_pixel_size = (250, 250)

# Define CPC concentrations for each class
cpc_concentration_map = {
    ('AF6', 'Day1', 'Online_Crop_Batch_1_X100'): 0.2819,
    ('AF6', 'Day1', 'Online_Crop_Batch_2_X100'): 0.2699,
    ('AF6', 'Day1', 'Online_Crop_Batch_3_X100'): 0.2813,
    ('AF6', 'Day4', 'Online_Crop_Batch_1_X100'): 0.3802,
    ('AF6', 'Day4', 'Online_Crop_Batch_2_X100'): 0.3826,
    ('AF6', 'Day4', 'Online_Crop_Batch_3_X100'): 0.3760,
    ('AF6', 'Day6', 'Online_Crop_Batch_1_X100'): 0.4081,
    ('AF6', 'Day6', 'Online_Crop_Batch_2_X100'): 0.4111,
    ('AF6', 'Day6', 'Online_Crop_Batch_3_X100'): 0.4040,
    ('AF6', 'Day8', 'Online_Crop_Batch_1_X100'): 0.4618,
    ('AF6', 'Day8', 'Online_Crop_Batch_2_X100'): 0.4690,
    ('AF6', 'Day8', 'Online_Crop_Batch_3_X100'): 0.4588,
    ('AF6', 'Day11', 'Online_Crop_Batch_1_X100'): 0.3905,
    ('AF6', 'Day11', 'Online_Crop_Batch_2_X100'): 0.4225,
    ('AF6', 'Day11', 'Online_Crop_Batch_3_X100'): 0.3248,
    ('AF6', 'Day14', 'Online_Crop_Batch_1_X100'): 0.4331,
    ('AF6', 'Day14', 'Online_Crop_Batch_2_X100'): 0.3277,
    ('AF6', 'Day14', 'Online_Crop_Batch_3_X100'): 0.4019,
    ('AF6', 'Day16', 'Online_Crop_Batch_1_X100'): 0.1529,
    ('AF6', 'Day16', 'Online_Crop_Batch_2_X100'): 0.1755,
    ('AF6', 'Day16', 'Online_Crop_Batch_3_X100'): 0.1813,
    ('Bg-11', 'Day1', 'Online_Crop_Batch_1_X100'): 0.2556,
    ('Bg-11', 'Day1', 'Online_Crop_Batch_2_X100'): 0.2039,
    ('Bg-11', 'Day1', 'Online_Crop_Batch_3_X100'): 0.2233,
    ('Bg-11', 'Day4', 'Online_Crop_Batch_1_X100'): 0.2609,
    ('Bg-11', 'Day4', 'Online_Crop_Batch_2_X100'): 0.3246,
    ('Bg-11', 'Day4', 'Online_Crop_Batch_3_X100'): 0.2879,
    ('Bg-11', 'Day6', 'Online_Crop_Batch_1_X100'): 0.7347,
    ('Bg-11', 'Day6', 'Online_Crop_Batch_2_X100'): 0.7862,
    ('Bg-11', 'Day6', 'Online_Crop_Batch_3_X100'): 0.7636,
    ('Bg-11', 'Day8', 'Online_Crop_Batch_1_X100'): 0.9611,
    ('Bg-11', 'Day8', 'Online_Crop_Batch_2_X100'): 0.9587,
    ('Bg-11', 'Day8', 'Online_Crop_Batch_3_X100'): 0.9785,
    ('Bg-11', 'Day11', 'Online_Crop_Batch_1_X100'): 0.7133,
    ('Bg-11', 'Day11', 'Online_Crop_Batch_2_X100'): 0.7277,
    ('Bg-11', 'Day11', 'Online_Crop_Batch_3_X100'): 0.7256,
    ('Bg-11', 'Day14', 'Online_Crop_Batch_1_X100'): 0.5051,
    ('Bg-11', 'Day14', 'Online_Crop_Batch_2_X100'): 0.5228,
    ('Bg-11', 'Day14', 'Online_Crop_Batch_3_X100'): 0.5239,
    ('Bg-11', 'Day16', 'Online_Crop_Batch_1_X100'): 0.2020,
    ('Bg-11', 'Day16', 'Online_Crop_Batch_2_X100'): 0.2451,
    ('Bg-11', 'Day16', 'Online_Crop_Batch_3_X100'): 0.2430,
    ('Zarrouk', 'Day1', 'Online_Crop_Batch_1_X100'): 0.2266,
    ('Zarrouk', 'Day1', 'Online_Crop_Batch_2_X100'): 0.1728,
    ('Zarrouk', 'Day1', 'Online_Crop_Batch_3_X100'): 0.1390,
    ('Zarrouk', 'Day4', 'Online_Crop_Batch_1_X100'): 0.2875,
    ('Zarrouk', 'Day4', 'Online_Crop_Batch_2_X100'): 0.3559,
    ('Zarrouk', 'Day4', 'Online_Crop_Batch_3_X100'): 0.2857,
    ('Zarrouk', 'Day6', 'Online_Crop_Batch_1_X100'): 0.5603,
    ('Zarrouk', 'Day6', 'Online_Crop_Batch_2_X100'): 0.5851,
    ('Zarrouk', 'Day6', 'Online_Crop_Batch_3_X100'): 0.5495,
    ('Zarrouk', 'Day8', 'Online_Crop_Batch_1_X100'): 0.8852,
    ('Zarrouk', 'Day8', 'Online_Crop_Batch_2_X100'): 0.7938,
    ('Zarrouk', 'Day8', 'Online_Crop_Batch_3_X100'): 0.7989,
    ('Zarrouk', 'Day11', 'Online_Crop_Batch_1_X100'): 0.7158,
    ('Zarrouk', 'Day11', 'Online_Crop_Batch_2_X100'): 0.7064,
    ('Zarrouk', 'Day11', 'Online_Crop_Batch_3_X100'): 0.7040,
    ('Zarrouk', 'Day14', 'Online_Crop_Batch_1_X100'): 0.5682,
    ('Zarrouk', 'Day14', 'Online_Crop_Batch_2_X100'): 0.5849,
    ('Zarrouk', 'Day14', 'Online_Crop_Batch_3_X100'): 0.5791,
    ('Zarrouk', 'Day16', 'Online_Crop_Batch_1_X100'): 0.3991,
    ('Zarrouk', 'Day16', 'Online_Crop_Batch_2_X100'): 0.2335,
    ('Zarrouk', 'Day16', 'Online_Crop_Batch_3_X100'): 0.4090,
}

# Load images and corresponding CPC concentrations
for medium_folder in media_folders:
    medium_dir = os.path.join(main_data_dir, medium_folder)
    for day_folder in os.listdir(medium_dir):
        for batch_folder in os.listdir(os.path.join(medium_dir, day_folder)):
            cpc_key = (medium_folder, day_folder, batch_folder)
            cpc_concentration = cpc_concentration_map.get(cpc_key, 0.0)  # Default to 0.0 if not found
            image_count = 0
            batch_path = os.path.join(medium_dir, day_folder, batch_folder)
            for img_file in os.listdir(batch_path):
                if image_count >= 200:  # Limit the number of images per batch
                    break
                img_path = os.path.join(batch_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    # Image loading failed; handle the error here
                    print(f"Error loading image: {img_path}")
                    continue
                if img.shape[:2] != desired_pixel_size:
                    img = cv2.resize(img, desired_pixel_size)
                img = img / 255.0
                images.append(img)
                cpc_concentrations.append(cpc_concentration)
                image_count += 1

X = np.array(images)
y = np.array(cpc_concentrations)

# Split the Data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the number of images for training and testing
print(f"Number of images for training: {len(X_train)}")
print(f"Number of images for testing: {len(X_test)}")

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=desired_pixel_size + (3,)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='linear')  # Output layer with 3 neurons for low, medium, and high
])

# Set a smaller learning rate (e.g., 0.001)
custom_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error', metrics=[RSquare(dtype=tf.float32, y_shape=(1,)), MeanAbsoluteError(), RootMeanSquaredError()])

# Define image data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=30,          # Random rotation between -30 and 30 degrees
    width_shift_range=0.2,      # Random horizontal shift
    height_shift_range=0.2,     # Random vertical shift
    shear_range=0.2,            # Shear intensity
    zoom_range=0.2,             # Random zoom
    horizontal_flip=True,       # Random horizontal flip
    vertical_flip=False,        # No vertical flip
    fill_mode='nearest'         # Fill mode for points outside the input boundary
)

# Train the Model with data augmentation
batch_size = 64
epochs = 100

# Create data generators for training and testing
train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)
test_datagen = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=True)

# Fit the model using the augmented data generator
history = model.fit(
    train_datagen,
    epochs=epochs,
    validation_data=test_datagen
)

print(history.history.keys())

# Advanced Visualizations
# Mean Squared Error vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Training MSE vs Validation MSE')
plt.show()

# Mean Squared Error vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(history.history['r_square'], label='Training R Square')
plt.plot(history.history['val_r_square'], label='Validation R Square')
plt.xlabel('Epochs')
plt.ylabel('R Square')
plt.legend()
plt.title('Training R Square vs Validation R Square')
plt.show()