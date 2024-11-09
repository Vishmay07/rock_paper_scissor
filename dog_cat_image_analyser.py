import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Set image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Cell 3: Create image generators for train, validation, and test datasets

# Create ImageDataGenerators with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up the data generators using flow_from_directory
train_data_gen = train_datagen.flow_from_directory(
    'cats_and_dogs/train',  # Directory for training data
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Resize images
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Binary classification (cat or dog)   
)

validation_data_gen = validation_datagen.flow_from_directory(
    'cats_and_dogs/validation',  # Directory for validation data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data_gen = test_datagen.flow_from_directory(
    'cats_and_dogs/test',  # Directory for test data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels for test data
    shuffle=False  # We need to keep the order for predictions
)

# Cell 4: Function to plot images
def plotImages(images, labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        if labels is not None:
            plt.title(f'{labels[i]}')
        plt.axis('off')
    plt.show()

# Get a batch of images and labels from the training set
images, labels = next(train_data_gen)

# Plot the images
plotImages(images)

# Cell 5: Data Augmentation to prevent overfitting
train_image_generator = ImageDataGenerator(
    rescale=1./255,  # Rescale the images
    rotation_range=40,  # Random rotation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill pixels after transformation
)

# Create the augmented data generator
train_data_gen = train_image_generator.flow_from_directory(
    'cats_and_dogs/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Cell 7: Build the CNN model
model = models.Sequential()

# First Convolutional Block
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Block
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Block
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fully Connected Block
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Cell 8: Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=10,  # You can adjust this value
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.samples // BATCH_SIZE
)

# Cell 9: Visualize accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot training and validation accuracy
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

# Plot training and validation loss
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Cell 10: Make predictions on the test set
predictions = model.predict(test_data_gen, steps=test_data_gen.samples // BATCH_SIZE, verbose=1)

# Convert predictions to 0 or 1 (cat or dog)
predictions = (predictions > 0.5).astype("int32")

# Plot the test images with predicted labels
plotImages(test_data_gen.next()[0], predictions)
