import pandas as pd
import numpy as np # linear algebra
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split
import shutil
import cv2
from tqdm import tqdm
import imutils

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """

    ## Get the shape
    shape = set_name[0].shape
    set_new = []
    for img in set_name:

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        new_img = cv2.resize(new_img, shape[:2])
        set_new.append(new_img)


    return np.array(set_new)


def load_data_tonumpy(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels



def create_train_test_dataset_from_generators(train_directory, test_directory):

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
