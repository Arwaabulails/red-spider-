from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator class from keras module

def load_data(train_dir, test_dir, img_size=(300, 300), batch_size=32, validation_split=0.1):
    """
    Load and preprocess data for training, validation, and testing.

    Args:
        train_dir (str): Path to the training dataset.
        test_dir (str): Path to the testing dataset.
        img_size (tuple): Target size for resizing images (width, height).
        batch_size (int): Number of images per batch.
        validation_split (float): Percentage of training data for validation.

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Data preprocessing for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load training data (90% of the training set)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Load validation data (10% of the training set)
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
