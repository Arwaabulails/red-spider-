import tensorflow as tf

from tensorflow.keras.applications import MobileNetV3Small

from tensorflow.keras import models, layers
from preprocessing import load_data

# Paths to dataset directories
train_dir = 'augmented_dataset_new/train_new'
test_dir = 'augmented_dataset_new/test_new'

# Parameters
img_size = (300, 300)
batch_size = 16
num_epochs = 10
validation_split = 0.1

# Load data using the data_preprocessing module
train_generator, val_generator, test_generator = load_data(
    train_dir=train_dir,
    test_dir=test_dir,
    img_size=img_size,
    batch_size=batch_size,
    validation_split=validation_split
)

# Load MobileNetV3 Small pre-trained on ImageNet
base_model = MobileNetV3Small(
    input_shape=(300, 300, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model for transfer learning

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with validation data
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')
