"""
Test Cases for Smart Vision Classifier
Unit tests for model inference and data processing
"""

import unittest
import numpy as np
import os
import sys
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import CNNClassifier
from data.preprocessor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for data preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cifar_processor = DataProcessor("cifar10")
        self.mnist_processor = DataProcessor("mnist")
    
    def test_cifar10_initialization(self):
        """Test CIFAR-10 processor initialization"""
        self.assertEqual(self.cifar_processor.dataset_type, "cifar10")
        self.assertEqual(self.cifar_processor.input_shape, (32, 32, 3))
        self.assertEqual(self.cifar_processor.num_classes, 10)
        self.assertEqual(len(self.cifar_processor.class_names), 10)
    
    def test_mnist_initialization(self):
        """Test MNIST processor initialization"""
        self.assertEqual(self.mnist_processor.dataset_type, "mnist")
        self.assertEqual(self.mnist_processor.input_shape, (28, 28, 1))
        self.assertEqual(self.mnist_processor.num_classes, 10)
        self.assertEqual(len(self.mnist_processor.class_names), 10)
    
    def test_load_data_shapes(self):
        """Test data loading returns correct shapes"""
        # Test CIFAR-10
        x_train, y_train, x_test, y_test = self.cifar_processor.load_data()
        self.assertEqual(x_train.shape[1:], (32, 32, 3))
        self.assertEqual(x_test.shape[1:], (32, 32, 3))
        self.assertGreater(len(x_train), 0)
        self.assertGreater(len(x_test), 0)
        
        # Test MNIST
        x_train, y_train, x_test, y_test = self.mnist_processor.load_data()
        self.assertEqual(x_train.shape[1:], (28, 28, 1))
        self.assertEqual(x_test.shape[1:], (28, 28, 1))
    
    def test_prepare_data(self):
        """Test data preparation with validation split"""
        data = self.cifar_processor.prepare_data(validation_split=0.2)
        
        # Check all required keys exist
        required_keys = ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']
        for key in required_keys:
            self.assertIn(key, data)
        
        # Check shapes
        self.assertEqual(data['x_train'].shape[1:], (32, 32, 3))
        self.assertEqual(data['y_train'].shape[1], 10)  # One-hot encoded
        
        # Check validation split
        total_train_val = len(data['x_train']) + len(data['x_val'])
        val_ratio = len(data['x_val']) / total_train_val
        self.assertAlmostEqual(val_ratio, 0.2, places=1)


class TestCNNModel(unittest.TestCase):
    """Test cases for CNN model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cifar_model = CNNClassifier("cifar10")
        self.mnist_model = CNNClassifier("mnist")
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.cifar_model.dataset_type, "cifar10")
        self.assertEqual(self.cifar_model.input_shape, (32, 32, 3))
        self.assertEqual(self.cifar_model.num_classes, 10)
        
        self.assertEqual(self.mnist_model.dataset_type, "mnist")
        self.assertEqual(self.mnist_model.input_shape, (28, 28, 1))
        self.assertEqual(self.mnist_model.num_classes, 10)
    
    def test_build_model(self):
        """Test model building"""
        model = self.cifar_model.build_model()
        
        # Check model exists
        self.assertIsNotNone(model)
        
        # Check input/output shapes
        self.assertEqual(model.input_shape[1:], (32, 32, 3))
        self.assertEqual(model.output_shape[1], 10)
        
        # Check model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_predict_shape(self):
        """Test prediction output shapes"""
        # Build model
        self.cifar_model.build_model()
        
        # Create dummy image
        dummy_image = np.random.random((32, 32, 3))
        
        # Make prediction
        predicted_class, confidence, probabilities = self.cifar_model.predict(dummy_image)
        
        # Check outputs
        self.assertIsInstance(predicted_class, str)
        self.assertIsInstance(confidence, (float, np.float32, np.float64))
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
    
    def test_model_summary(self):
        """Test model summary generation"""
        summary = self.cifar_model.get_model_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)


class TestImageProcessing(unittest.TestCase):
    """Test cases for image preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor("cifar10")
        
        # Create test images
        self.test_image_rgb = Image.new('RGB', (64, 64), color='red')
        self.test_image_gray = Image.new('L', (64, 64), color=128)
    
    def test_preprocess_rgb_image(self):
        """Test preprocessing RGB image"""
        # Save test image temporarily
        test_path = "test_rgb.png"
        self.test_image_rgb.save(test_path)
        
        try:
            processed = self.processor.preprocess_image(test_path)
            
            # Check shape and type
            self.assertEqual(processed.shape, (32, 32, 3))
            self.assertEqual(processed.dtype, np.float32)
            
            # Check normalization (values should be in [0, 1])
            self.assertTrue(np.all(processed >= 0))
            self.assertTrue(np.all(processed <= 1))
            
        finally:
            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)
    
    def test_invalid_image_path(self):
        """Test error handling for invalid image path"""
        with self.assertRaises(ValueError):
            self.processor.preprocess_image("nonexistent_image.jpg")


class TestModelIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        # Initialize components
        processor = DataProcessor("cifar10")
        model = CNNClassifier("cifar10")
        
        # Build model
        model.build_model()
        
        # Create test image
        test_image = np.random.random((32, 32, 3))
        
        # Make prediction
        predicted_class, confidence, probabilities = model.predict(test_image)
        
        # Verify outputs
        self.assertIn(predicted_class, processor.class_names)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        self.assertEqual(len(probabilities), 10)
    
    def test_data_flow_consistency(self):
        """Test data flow consistency between processor and model"""
        processor = DataProcessor("mnist")
        model = CNNClassifier("mnist")
        
        # Check shape consistency
        self.assertEqual(processor.input_shape, model.input_shape)
        self.assertEqual(processor.num_classes, model.num_classes)
        self.assertEqual(processor.class_names, model.class_names)


class TestPerformance(unittest.TestCase):
    """Performance and memory tests"""
    
    def test_prediction_speed(self):
        """Test prediction performance"""
        import time
        
        model = CNNClassifier("cifar10")
        model.build_model()
        
        # Test single prediction
        test_image = np.random.random((32, 32, 3))
        
        start_time = time.time()
        model.predict(test_image)
        prediction_time = time.time() - start_time
        
        # Prediction should be fast (less than 1 second)
        self.assertLess(prediction_time, 1.0)
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        model = CNNClassifier("cifar10")
        model.build_model()
        
        # Test batch of images
        batch_images = np.random.random((5, 32, 32, 3))
        
        # Predict each image in batch
        predictions = []
        for image in batch_images:
            pred_class, confidence, probs = model.predict(image)
            predictions.append((pred_class, confidence))
        
        self.assertEqual(len(predictions), 5)


def run_tests():
    """Run all tests"""
    print("🧪 Running Smart Vision Classifier Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataProcessor,
        TestCNNModel,
        TestImageProcessing,
        TestModelIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
