"""
Smart Vision Classifier - Main Entry Point
Run this file to start the application
"""

import os
import sys
import subprocess
import argparse


def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)


def run_streamlit_app():
    """Launch the Streamlit application"""
    print("🚀 Starting Smart Vision Classifier...")
    
    # Change to app directory
    app_path = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ App file not found: {app_path}")
        sys.exit(1)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")


def train_model_cli(dataset="cifar10", epochs=20, batch_size=32):
    """Train model via command line interface"""
    print(f"🎯 Training {dataset} model...")
    
    try:
        from models.cnn_model import CNNClassifier
        from data.preprocessor import DataProcessor
        
        # Initialize components
        processor = DataProcessor(dataset)
        model = CNNClassifier(dataset)
        
        print("📊 Loading and preparing data...")
        data = processor.prepare_data()
        
        print("🏗️ Building model...")
        model.build_model()
        
        print(f"🚀 Starting training for {epochs} epochs...")
        history = model.train(
            data['x_train'], data['y_train'],
            data['x_val'], data['y_val'],
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save model
        model_path = f"models/trained_{dataset}_model.h5"
        model.save_model(model_path)
        
        # Evaluate
        print("📊 Evaluating model...")
        test_results = model.evaluate(data['x_test'], data['y_test'])
        
        print("\n✅ Training completed!")
        print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"Test Loss: {test_results['test_loss']:.4f}")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Smart Vision Classifier")
    
    parser.add_argument(
        "--mode", 
        choices=["app", "train", "install"],
        default="app",
        help="Mode to run: app (Streamlit UI), train (CLI training), or install (install requirements)"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "mnist"],
        default="cifar10",
        help="Dataset to use for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    # Display banner
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🔍 SMART VISION CLASSIFIER 🔍                   ║
    ║                                                               ║
    ║          AI-Powered Image Classification System               ║
    ║              Built with TensorFlow & Streamlit               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if args.mode == "install":
        install_requirements()
    elif args.mode == "train":
        train_model_cli(args.dataset, args.epochs, args.batch_size)
    elif args.mode == "app":
        run_streamlit_app()
    else:
        print("❌ Invalid mode specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
