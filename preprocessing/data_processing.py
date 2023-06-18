import pandas as pd
import numpy as np # linear algebra
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow import keras



def create_train_test_dataset(train_directory, test_directory):

    img_height, img_width = 224,224
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    color_mode='grayscale',
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    interpolation='nearest',
    batch_size=batch_size)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
    test_directory,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    interpolation='nearest',
    batch_size=batch_size)

    return train_ds, valid_ds


if __name__ == '__main__':
    # Set the source directory containing the "yes" and "no" folders
    source_directory = 'D:/Projects/tumai_braintumor_classification/dataset/brain_tumor_dataset'

    # Set the destination directory for train and test data
    destination_directory = 'D:/Projects/tumai_braintumor_classification/dataset'

    # Set the train and test data ratios
    test_size = 0.2

    # Create the train and test data directories
    train_directory = os.path.join(destination_directory, 'train')
    test_directory = os.path.join(destination_directory, 'test')

    os.makedirs(train_directory+"/yes", exist_ok=True)
    os.makedirs(train_directory+"/no", exist_ok=True)
    os.makedirs(test_directory+"/yes", exist_ok=True)
    os.makedirs(test_directory+"/no", exist_ok=True)

    # Get the list of image files from the "yes" folder
    yes_images_directory = os.path.join(source_directory, 'yes')
    yes_image_files = [os.path.join(yes_images_directory, file) for file in os.listdir(yes_images_directory)]

    # Get the list of image files from the "no" folder
    no_images_directory = os.path.join(source_directory, 'no')
    no_image_files = [os.path.join(no_images_directory, file) for file in os.listdir(no_images_directory)]

    # Split the image files into train and test sets
    train_yes_files, test_yes_files = train_test_split(yes_image_files, test_size=test_size, random_state=42)
    train_no_files, test_no_files = train_test_split(no_image_files, test_size=test_size, random_state=42)

    # Move the "yes" train images to the train directory
    for file in train_yes_files:
        filename = os.path.basename(file)
        destination_path = os.path.join(train_directory, 'yes', filename)
        shutil.copyfile(file, destination_path)

    # Move the "no" train images to the train directory
    for file in train_no_files:
        filename = os.path.basename(file)
        destination_path = os.path.join(train_directory, 'no', filename)
        shutil.copyfile(file, destination_path)

    # Move the "yes" test images to the test directory
    for file in test_yes_files:
        filename = os.path.basename(file)
        destination_path = os.path.join(test_directory, 'yes', filename)
        shutil.copyfile(file, destination_path)

    # Move the "no" test images to the test directory
    for file in test_no_files:
        filename = os.path.basename(file)
        destination_path = os.path.join(test_directory, 'no', filename)
        shutil.copyfile(file, destination_path)

    print("Data splitting completed!")
