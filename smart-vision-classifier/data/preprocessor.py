"""
Data Preprocessing Module
Handles loading and preprocessing of MNIST and CIFAR-10 datasets
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, Dict
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class DataProcessor:
    """
    Data preprocessing class for image classification datasets
    """
    
    def __init__(self, dataset_type: str = "cifar10"):
        """
        Initialize data processor
        
        Args:
            dataset_type (str): Type of dataset ('mnist' or 'cifar10')
        """
        self.dataset_type = dataset_type.lower()
        
        if self.dataset_type == "mnist":
            self.input_shape = (28, 28, 1)
            self.num_classes = 10
            self.class_names = [str(i) for i in range(10)]
        elif self.dataset_type == "cifar10":
            self.input_shape = (32, 32, 3)
            self.num_classes = 10
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            raise ValueError("Unsupported dataset type. Use 'mnist' or 'cifar10'")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the dataset
        
        Returns:
            tuple: (x_train, y_train, x_test, y_test)
        """
        if self.dataset_type == "mnist":
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            # Reshape and normalize
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            
        elif self.dataset_type == "cifar10":
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            
            # Normalize pixel values
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Flatten labels
            y_train = y_train.flatten()
            y_test = y_test.flatten()
        
        return x_train, y_train, x_test, y_test
    
    def prepare_data(self, validation_split: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepare training, validation, and test datasets
        
        Args:
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            dict: Dictionary containing all data splits
        """
        x_train, y_train, x_test, y_test = self.load_data()
        
        # Create validation split
        val_size = int(len(x_train) * validation_split)
        indices = np.random.permutation(len(x_train))
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single image for prediction
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB' and self.dataset_type == "cifar10":
                image = image.convert('RGB')
            elif image.mode != 'L' and self.dataset_type == "mnist":
                image = image.convert('L')
            
            # Resize to target size
            target_size = self.input_shape[:2]
            image = image.resize(target_size)
            
            # Convert to array
            image_array = np.array(image)
            
            # Reshape for MNIST (add channel dimension)
            if self.dataset_type == "mnist":
                if len(image_array.shape) == 2:
                    image_array = image_array.reshape(28, 28, 1)
            
            # Normalize
            image_array = image_array.astype('float32') / 255.0
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def preprocess_uploaded_image(self, uploaded_file) -> np.ndarray:
        """
        Preprocess an uploaded image from Streamlit
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load image from uploaded file
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB' and self.dataset_type == "cifar10":
                image = image.convert('RGB')
            elif image.mode != 'L' and self.dataset_type == "mnist":
                image = image.convert('L')
            
            # Resize to target size
            target_size = self.input_shape[:2]
            image = image.resize(target_size)
            
            # Convert to array
            image_array = np.array(image)
            
            # Reshape for MNIST (add channel dimension)
            if self.dataset_type == "mnist":
                if len(image_array.shape) == 2:
                    image_array = image_array.reshape(28, 28, 1)
            
            # Normalize
            image_array = image_array.astype('float32') / 255.0
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error processing uploaded image: {str(e)}")
    
    def display_sample_images(self, num_samples: int = 10) -> plt.Figure:
        """
        Display sample images from the dataset
        
        Args:
            num_samples (int): Number of sample images to display
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        x_train, y_train, _, _ = self.load_data()
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle(f'Sample Images from {self.dataset_type.upper()} Dataset', fontsize=16)
        
        for i in range(num_samples):
            row = i // 5
            col = i % 5
            
            if self.dataset_type == "mnist":
                axes[row, col].imshow(x_train[i].reshape(28, 28), cmap='gray')
            else:
                axes[row, col].imshow(x_train[i])
            
            axes[row, col].set_title(f'Class: {self.class_names[y_train[i]]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution of the dataset
        
        Returns:
            dict: Class distribution
        """
        _, y_train, _, y_test = self.load_data()
        
        train_dist = {self.class_names[i]: np.sum(y_train == i) for i in range(self.num_classes)}
        test_dist = {self.class_names[i]: np.sum(y_test == i) for i in range(self.num_classes)}
        
        return {
            'train_distribution': train_dist,
            'test_distribution': test_dist
        }
