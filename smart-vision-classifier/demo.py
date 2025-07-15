"""
Quick Demo Script for Smart Vision Classifier
This script demonstrates the key functionality without the UI
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_model import CNNClassifier
from data.preprocessor import DataProcessor


def create_sample_image(dataset_type="cifar10"):
    """Create a sample image for testing"""
    if dataset_type == "cifar10":
        # Create a simple colored image
        image = np.zeros((32, 32, 3))
        image[8:24, 8:24, 0] = 1.0  # Red square
        return image
    else:  # mnist
        # Create a simple digit-like pattern
        image = np.zeros((28, 28, 1))
        image[10:18, 10:18, 0] = 1.0  # White square
        return image


def demo_data_loading():
    """Demonstrate data loading functionality"""
    print("🔥 Smart Vision Classifier Demo")
    print("=" * 50)
    
    print("\n📊 Testing Data Loading...")
    
    # Test CIFAR-10
    print("Loading CIFAR-10 dataset...")
    processor = DataProcessor("cifar10")
    x_train, y_train, x_test, y_test = processor.load_data()
    
    print(f"✅ CIFAR-10 loaded successfully!")
    print(f"   Training images: {x_train.shape}")
    print(f"   Test images: {x_test.shape}")
    print(f"   Classes: {processor.class_names}")
    
    # Test MNIST
    print("\nLoading MNIST dataset...")
    processor_mnist = DataProcessor("mnist")
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = processor_mnist.load_data()
    
    print(f"✅ MNIST loaded successfully!")
    print(f"   Training images: {x_train_mnist.shape}")
    print(f"   Test images: {x_test_mnist.shape}")
    
    return processor, processor_mnist


def demo_model_building():
    """Demonstrate model building"""
    print("\n🏗️ Testing Model Building...")
    
    # Build CIFAR-10 model
    model = CNNClassifier("cifar10")
    built_model = model.build_model()
    
    print("✅ CIFAR-10 model built successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Number of classes: {model.num_classes}")
    print(f"   Total parameters: {built_model.count_params():,}")
    
    return model


def demo_prediction():
    """Demonstrate prediction functionality"""
    print("\n🔍 Testing Prediction...")
    
    # Create model
    model = CNNClassifier("cifar10")
    model.build_model()
    
    # Create sample image
    sample_image = create_sample_image("cifar10")
    
    print(f"Sample image shape: {sample_image.shape}")
    
    # Make prediction
    predicted_class, confidence, probabilities = model.predict(sample_image)
    
    print(f"✅ Prediction completed!")
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Top 3 probabilities:")
    
    # Show top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    for i, idx in enumerate(top_indices):
        class_name = model.class_names[idx]
        prob = probabilities[idx]
        print(f"   {i+1}. {class_name}: {prob:.2%}")


def demo_training_small():
    """Demonstrate training with a small subset"""
    print("\n🎯 Testing Training (Small Subset)...")
    
    try:
        # Initialize components
        processor = DataProcessor("mnist")  # Use MNIST for faster training
        model = CNNClassifier("mnist")
        
        print("Loading data...")
        data = processor.prepare_data(validation_split=0.2)
        
        # Use only a small subset for demo
        subset_size = 1000
        x_train_small = data['x_train'][:subset_size]
        y_train_small = data['y_train'][:subset_size]
        x_val_small = data['x_val'][:200]
        y_val_small = data['y_val'][:200]
        
        print(f"Training on {subset_size} samples for 3 epochs (demo)...")
        
        # Train for just a few epochs
        history = model.train(
            x_train_small, y_train_small,
            x_val_small, y_val_small,
            epochs=3,
            batch_size=32
        )
        
        print("✅ Training completed!")
        print(f"   Final training accuracy: {history['accuracy'][-1]:.4f}")
        print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        
        # Test prediction on trained model
        test_image = data['x_test'][0]
        predicted_class, confidence, _ = model.predict(test_image)
        actual_class = model.class_names[np.argmax(data['y_test'][0])]
        
        print(f"   Test prediction: {predicted_class} (actual: {actual_class})")
        print(f"   Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"❌ Training demo failed: {str(e)}")
        print("   This is normal if you don't have enough memory or time")


def demo_image_preprocessing():
    """Demonstrate image preprocessing"""
    print("\n🖼️ Testing Image Preprocessing...")
    
    processor = DataProcessor("cifar10")
    
    # Create a test image
    test_image = Image.new('RGB', (64, 64), color='red')
    test_path = "temp_test_image.png"
    
    try:
        test_image.save(test_path)
        
        # Preprocess the image
        processed = processor.preprocess_image(test_path)
        
        print("✅ Image preprocessing completed!")
        print(f"   Original size: 64x64")
        print(f"   Processed shape: {processed.shape}")
        print(f"   Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Image preprocessing failed: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)


def run_full_demo():
    """Run the complete demonstration"""
    print("🚀 Starting Smart Vision Classifier Demo")
    print("This will test all major components without the UI")
    print("=" * 60)
    
    try:
        # Run all demo functions
        processor, processor_mnist = demo_data_loading()
        model = demo_model_building()
        demo_prediction()
        demo_image_preprocessing()
        
        # Ask user if they want to run training demo
        print("\n" + "=" * 60)
        response = input("🤔 Run training demo? (takes ~2-3 minutes) [y/N]: ")
        
        if response.lower() in ['y', 'yes']:
            demo_training_small()
        else:
            print("⏭️ Skipping training demo")
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python main.py' to start the web interface")
        print("2. Or run 'python main.py --mode train' to train a full model")
        print("3. Check the README.md for more details")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please check your Python environment and dependencies")


if __name__ == "__main__":
    run_full_demo()
