"""
Configuration settings for Smart Vision Classifier
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# Model settings
DEFAULT_DATASET = "cifar10"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2

# CIFAR-10 configuration
CIFAR10_CONFIG = {
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "class_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
}

# MNIST configuration
MNIST_CONFIG = {
    "input_shape": (28, 28, 1),
    "num_classes": 10,
    "class_names": [str(i) for i in range(10)]
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Smart Vision Classifier",
    "page_icon": "🔍",
    "layout": "wide",
    "theme": {
        "primaryColor": "#667eea",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730"
    }
}

# Training configuration
TRAINING_CONFIG = {
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.2,
    "min_lr": 1e-7,
    "callbacks": ["early_stopping", "reduce_lr", "model_checkpoint"]
}

# Model architecture configuration
MODEL_CONFIG = {
    "conv_layers": [
        {"filters": 32, "kernel_size": (3, 3), "activation": "relu"},
        {"filters": 64, "kernel_size": (3, 3), "activation": "relu"},
        {"filters": 128, "kernel_size": (3, 3), "activation": "relu"}
    ],
    "dense_layers": [
        {"units": 512, "activation": "relu", "dropout": 0.5},
        {"units": 256, "activation": "relu", "dropout": 0.5}
    ],
    "batch_normalization": True,
    "dropout_conv": 0.25,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy", "top_k_categorical_accuracy"]
}

# Data augmentation settings
DATA_AUGMENTATION = {
    "cifar10": {
        "rotation_range": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "horizontal_flip": True,
        "zoom_range": 0.1
    },
    "mnist": {
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "zoom_range": 0.1
    }
}

# UI Colors and Styling
UI_COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#28a745",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40"
}
