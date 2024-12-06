import zipfile
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, extract_path="../data", target_size=(128, 128)):
        """
        Initialize the Preprocessor with a default extraction path and target image size.

        Parameters:
            extract_path (str): Directory where the dataset will be extracted. Default is "../data".
            target_size (tuple): Dimensions to resize the images to. Default is (128, 128).
        """
        self.extract_path = extract_path
        self.target_size = target_size
        os.makedirs(self.extract_path, exist_ok=True)

    def load_data(self, zip_path):
        """
        Extracts a zip file containing the image dataset and organizes the folder structure.

        Parameters:
            zip_path (str): Path to the zip file containing the dataset.

        Returns:
            dict: A dictionary representing the folder structure, with directories as keys and file lists as values.
        """
        try:
            # Check if the file exists
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"The file '{zip_path}' does not exist.")

            # Extract the dataset
            print(f"\nExtracting the dataset from '{zip_path}' to '{self.extract_path}'...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.extract_path)
            print("Dataset extracted successfully!")

            # Generate the folder structure
            print("\nExtracted Folder Structure:")
            folder_structure = {}
            for root, dirs, files in os.walk(self.extract_path):
                level = root.replace(self.extract_path, "").count(os.sep)
                indent = " " * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                
                # Build folder structure as a dictionary
                folder_structure[os.path.basename(root)] = files

            return folder_structure

        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}")
        except zipfile.BadZipFile:
            print("Error: The provided zip file is invalid or corrupted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def resize_images(self, dataset_path):
        """
        Load images from the directory structure and resize them to a fixed dimension.

        Returns:
            np.array: Array of resized image data.
            np.array: Array of corresponding labels.
        """
        print("Loading and resizing images...")
        images, labels = [], []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add more valid extensions if needed
        
        for class_label in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_label)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if img_file.lower().endswith(valid_extensions):  # Check file extension
                        try:
                            img = load_img(img_path, target_size=self.target_size)
                            img_array = img_to_array(img)
                            images.append(img_array)
                            labels.append(class_label)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                    else:
                        print(f"Skipping non-image file: {img_path}")
        return np.array(images), np.array(labels)

    def normalize_images(self, images):
        """
        Normalize pixel values to the range [0, 1].

        Parameters:
            images (np.array): Array of image data.

        Returns:
            np.array: Normalized image data.
        """
        print("Normalizing images...")
        return images / 255.0

    def split_data(self, images, labels, test_size=0.3, val_size=0.2):
        """
        Split the dataset into training, validation, and test sets.

        Parameters:
            images (np.array): Array of image data.
            labels (np.array): Array of labels.
            test_size (float): Fraction of the data to use for testing.
            val_size (float): Fraction of the training data to use for validation.

        Returns:
            tuple: Split data (train_images, val_images, test_images, train_labels, val_labels, test_labels).
        """
        print("Splitting data into training, validation, and test sets...")
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, stratify=labels)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, stratify=y_train)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def check_class_balance(self, labels):
        """
        Check for class balance in the dataset.

        Parameters:
            labels (np.array): Array of class labels.

        Returns:
            dict: Counts of each class.
        """
        print("Checking class balance...")
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        for cls, count in class_counts.items():
            print(f"{cls}: {count}")
        return class_counts

    def augment_data(self):
        """
        Create an ImageDataGenerator for data augmentation.

        Returns:
            ImageDataGenerator: Configured data augmentation generator.
        """
        print("Creating data augmentation generator...")
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )