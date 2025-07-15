"""
CNN Model Architecture for Image Classification
Author: Smart Vision Classifier Team
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict, Any
import os


class CNNClassifier:
    """
    Convolutional Neural Network for image classification
    Supports both MNIST and CIFAR-10 datasets
    """
    
    def __init__(self, dataset_type: str = "cifar10", input_shape: Tuple = None):
        """
        Initialize the CNN classifier
        
        Args:
            dataset_type (str): Type of dataset ('mnist' or 'cifar10')
            input_shape (tuple): Shape of input images
        """
        self.dataset_type = dataset_type.lower()
        self.model = None
        self.history = None
        
        if input_shape is None:
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
        else:
            self.input_shape = input_shape
    
    def build_model(self) -> keras.Model:
        """
        Build the CNN model architecture
        
        Returns:
            keras.Model: Compiled CNN model
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 20, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the CNN model
        
        Args:
            x_train: Training images
            y_train: Training labels (one-hot encoded)
            x_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation for better generalization
        if self.dataset_type == "cifar10":
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            )
            datagen.fit(x_train)
            
            self.history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history.history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            x_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        test_loss, test_accuracy, test_top5_acc = self.model.evaluate(
            x_test, y_test, verbose=0
        )
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top5_accuracy': test_top5_acc
        }
    
    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Make prediction on a single image
        
        Args:
            image: Input image array
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            self.build_model()
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
