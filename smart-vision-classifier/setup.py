"""
Setup script for Smart Vision Classifier
Run this to set up the project environment
"""

import subprocess
import sys
import os
import platform


def print_banner():
    """Print setup banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🔍 SMART VISION CLASSIFIER 🔍                   ║
    ║                                                               ║
    ║                    Setup Assistant                            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_pip():
    """Check if pip is available"""
    print("🔍 Checking pip...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False


def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python packages...")
    print("   This may take a few minutes...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL)
        
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "models",
        "data",
        "static",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directories created successfully!")


def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import tensorflow as tf
        import streamlit as st
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        from PIL import Image
        
        print("✅ All packages imported successfully!")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   Streamlit version: {st.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False


def run_demo():
    """Ask user if they want to run the demo"""
    print("\n" + "="*60)
    response = input("🤔 Would you like to run a quick demo? [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        print("🚀 Running demo...")
        try:
            subprocess.run([sys.executable, "demo.py"])
        except KeyboardInterrupt:
            print("\n⚠️ Demo interrupted")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    else:
        print("⏭️ Skipping demo")


def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("   1. Run the web app:     python main.py")
    print("   2. Train a model:       python main.py --mode train")
    print("   3. Run tests:           python tests/test_model.py")
    print("   4. Quick demo:          python demo.py")
    print("\n📖 Documentation:")
    print("   - README.md: Full project documentation")
    print("   - config.py: Configuration options")
    print("\n🌐 Web Interface:")
    print("   - URL: http://localhost:8501")
    print("   - The app will open automatically in your browser")
    print("\n🐳 Docker (optional):")
    print("   - Build: docker build -t smart-vision-classifier .")
    print("   - Run:   docker run -p 8501:8501 smart-vision-classifier")


def main():
    """Main setup function"""
    print_banner()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup project
    if not install_requirements():
        sys.exit(1)
    
    create_directories()
    
    if not test_installation():
        print("\n⚠️ Installation test failed. You may need to install packages manually.")
    
    # Optional demo
    run_demo()
    
    # Show next steps
    print_next_steps()
    
    print("\n💡 Tip: If you encounter any issues, check the README.md for troubleshooting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check your Python environment and try again")
