import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil


def create_train_test_dataset(train_directory, test_directory):

    ## Create a train generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
        )
    
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        train_directory,
        # target_size=(224, 224),
        # batch_size=32,
        # class_mode='categorical',
        # subset='training',
        # shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        test_directory,
        # target_size=(224, 224),
        # batch_size=32,
        # class_mode='categorical',
        # subset='validation',
        # shuffle=True,
    )

    return train_generator, test_generator


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
