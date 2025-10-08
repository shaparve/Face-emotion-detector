import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
from tqdm import tqdm
import requests
import zipfile


class FER2013DataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.image_size = (48, 48)
        self.num_classes = 7
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def download_fer2013(self):
        csv_path = os.path.join(self.data_dir, 'fer2013.csv')
        if not os.path.exists(csv_path):
            print("Please download FER2013 dataset from Kaggle:")
            print("https://www.kaggle.com/datasets/msambare/fer2013")
            print(f"And place fer2013.csv in {self.data_dir}")
            return False
        return True

    def load_data(self, subset_size=None):
        csv_path = os.path.join(self.data_dir, 'fer2013.csv')

        if not os.path.exists(csv_path):
            if not self.download_fer2013():
                raise FileNotFoundError(f"FER2013 dataset not found at {csv_path}")

        print("Loading FER2013 dataset...")
        data = pd.read_csv(csv_path)

        if subset_size:
            data = data.sample(n=min(subset_size, len(data)), random_state=42)

        emotions = data['emotion'].values
        pixels = data['pixels'].values

        images = []
        labels = []

        print("Processing images...")
        for i in tqdm(range(len(pixels))):
            pixel_array = np.fromstring(pixels[i], dtype=int, sep=' ')
            image = pixel_array.reshape(48, 48).astype('float32')

            image = image / 255.0

            images.append(image)
            labels.append(emotions[i])

        images = np.array(images)
        images = np.expand_dims(images, -1)
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels

    def prepare_train_val_test(self, test_size=0.2, val_size=0.1):
        images, labels = self.load_data()

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=np.argmax(labels, axis=1)
        )

        val_split = val_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=val_split, random_state=42,
            stratify=np.argmax(y_train_val, axis=1)
        )

        print(f"Training set: {x_train.shape[0]} samples")
        print(f"Validation set: {x_val.shape[0]} samples")
        print(f"Test set: {x_test.shape[0]} samples")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def augment_data(self, images, labels, augmentation_factor=2):
        augmented_images = []
        augmented_labels = []

        for img, label in zip(images, labels):
            augmented_images.append(img)
            augmented_labels.append(label)

            for _ in range(augmentation_factor - 1):
                augmented = self._augment_image(img)
                augmented_images.append(augmented)
                augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)

    def _augment_image(self, image):
        img = image.copy()

        if np.random.random() > 0.5:
            img = np.fliplr(img)

        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((24, 24), angle, 1)
        img = cv2.warpAffine(img, M, (48, 48))

        shift_x = np.random.randint(-3, 3)
        shift_y = np.random.randint(-3, 3)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (48, 48))

        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 1)

        return img.reshape(48, 48, 1)