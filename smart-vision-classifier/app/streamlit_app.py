"""
Smart Vision Classifier - Streamlit Web Application
Beautiful interface for image classification using TensorFlow CNN
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import CNNClassifier
from data.preprocessor import DataProcessor


# Page configuration
st.set_page_config(
    page_title="Smart Vision Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .tech-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_processor(dataset_type="cifar10"):
    """Load the trained model and data processor"""
    try:
        processor = DataProcessor(dataset_type)
        model = CNNClassifier(dataset_type)
        
        # Check if pre-trained model exists
        model_path = f"models/trained_{dataset_type}_model.h5"
        if os.path.exists(model_path):
            model.load_model(model_path)
            return model, processor, True
        else:
            return model, processor, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False


def train_model_interface(model, processor, dataset_type):
    """Interface for training the model"""
    st.subheader("🚀 Train New Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", 5, 50, 20)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with col2:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
    if st.button("🎯 Start Training", type="primary"):
        with st.spinner("Training model... This may take a while."):
            try:
                # Prepare data
                data = processor.prepare_data(validation_split)
                
                # Create progress placeholder
                progress_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Train model
                history = model.train(
                    data['x_train'], data['y_train'],
                    data['x_val'], data['y_val'],
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                # Save trained model
                model_path = f"models/trained_{dataset_type}_model.h5"
                model.save_model(model_path)
                
                # Display training results
                st.success("✅ Model trained successfully!")
                
                # Plot training history
                fig = plot_training_history(history)
                st.plotly_chart(fig, use_container_width=True)
                
                # Evaluate on test data
                test_results = model.evaluate(data['x_test'], data['y_test'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{test_results['test_accuracy']:.4f}")
                with col2:
                    st.metric("Test Loss", f"{test_results['test_loss']:.4f}")
                with col3:
                    st.metric("Top-5 Accuracy", f"{test_results['test_top5_accuracy']:.4f}")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")


def plot_training_history(history):
    """Plot training history with Plotly"""
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig = go.Figure()
    
    # Training accuracy
    fig.add_trace(go.Scatter(
        x=list(epochs), y=history['accuracy'],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='#667eea', width=3)
    ))
    
    # Validation accuracy
    fig.add_trace(go.Scatter(
        x=list(epochs), y=history['val_accuracy'],
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='#764ba2', width=3)
    ))
    
    fig.update_layout(
        title='Model Training History',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def prediction_interface(model, processor):
    """Interface for making predictions"""
    st.subheader("🔍 Image Classification")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image for classification",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            try:
                # Preprocess image
                processed_image = processor.preprocess_uploaded_image(uploaded_file)
                
                # Make prediction
                predicted_class, confidence, all_probabilities = model.predict(processed_image)
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>🎯 Prediction Result</h3>
                    <p><strong>Predicted Class:</strong> {predicted_class}</p>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show top 5 predictions
                st.subheader("📊 Top 5 Predictions")
                
                # Create DataFrame for top predictions
                top_indices = np.argsort(all_probabilities)[::-1][:5]
                pred_data = []
                
                for i, idx in enumerate(top_indices):
                    pred_data.append({
                        'Rank': i + 1,
                        'Class': processor.class_names[idx],
                        'Probability': all_probabilities[idx],
                        'Percentage': f"{all_probabilities[idx]:.2%}"
                    })
                
                df = pd.DataFrame(pred_data)
                
                # Create horizontal bar chart
                fig = px.bar(
                    df, 
                    x='Probability', 
                    y='Class',
                    orientation='h',
                    title='Prediction Confidence Scores',
                    color='Probability',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")


def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Smart Vision Classifier</h1>
        <p>AI-Powered Image Classification with TensorFlow & CNN</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Choose Dataset",
        ["cifar10", "mnist"],
        help="Select the dataset for classification"
    )
    
    # Tech stack info
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <h4>🛠️ Tech Stack</h4>
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">Keras</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">OpenCV</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and processor
    model, processor, is_trained = load_model_and_processor(dataset_type)
    
    if model is None:
        st.error("Failed to load model. Please check the setup.")
        return
    
    # Display dataset info
    st.sidebar.markdown(f"""
    <div class="sidebar-info">
        <h4>📊 Dataset Info</h4>
        <p><strong>Type:</strong> {dataset_type.upper()}</p>
        <p><strong>Classes:</strong> {len(processor.class_names)}</p>
        <p><strong>Input Shape:</strong> {processor.input_shape}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    if is_trained:
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "🎯 Train Model", "📊 Dataset", "ℹ️ About"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Train Model", "📊 Dataset", "🔍 Classify", "ℹ️ About"])
        st.warning("⚠️ No pre-trained model found. Please train a model first.")
    
    with tab1:
        if is_trained:
            prediction_interface(model, processor)
        else:
            train_model_interface(model, processor, dataset_type)
    
    with tab2:
        if is_trained:
            train_model_interface(model, processor, dataset_type)
        else:
            # Dataset exploration
            st.subheader("📊 Dataset Exploration")
            
            if st.button("Show Sample Images"):
                with st.spinner("Loading sample images..."):
                    fig = processor.display_sample_images()
                    st.pyplot(fig)
            
            # Class distribution
            if st.button("Show Class Distribution"):
                dist = processor.get_class_distribution()
                
                # Create distribution chart
                train_dist = dist['train_distribution']
                fig = px.bar(
                    x=list(train_dist.keys()),
                    y=list(train_dist.values()),
                    title="Training Data Class Distribution",
                    labels={'x': 'Classes', 'y': 'Number of Samples'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if is_trained:
            # Dataset exploration
            st.subheader("📊 Dataset Exploration")
            
            if st.button("Show Sample Images"):
                with st.spinner("Loading sample images..."):
                    fig = processor.display_sample_images()
                    st.pyplot(fig)
            
            # Class distribution
            if st.button("Show Class Distribution"):
                dist = processor.get_class_distribution()
                
                # Create distribution chart
                train_dist = dist['train_distribution']
                fig = px.bar(
                    x=list(train_dist.keys()),
                    y=list(train_dist.values()),
                    title="Training Data Class Distribution",
                    labels={'x': 'Classes', 'y': 'Number of Samples'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            prediction_interface(model, processor)
    
    with tab4:
        st.subheader("ℹ️ About Smart Vision Classifier")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        Smart Vision Classifier is an advanced image classification system built with TensorFlow and Streamlit. 
        It uses Convolutional Neural Networks (CNN) to classify images from popular datasets like CIFAR-10 and MNIST.
        
        ### ✨ Features
        
        - **Real-time Image Classification**: Upload images and get instant predictions
        - **Interactive Training**: Train custom models with real-time progress tracking
        - **Beautiful Visualizations**: Dynamic charts showing training progress and prediction confidence
        - **Multiple Datasets**: Support for CIFAR-10 and MNIST datasets
        - **Model Persistence**: Save and load trained models
        - **Responsive Design**: Modern, mobile-friendly interface
        
        ### 🚀 Getting Started
        
        1. Choose your dataset (CIFAR-10 or MNIST)
        2. Train a new model or use a pre-trained one
        3. Upload images for classification
        4. Explore the dataset and model performance
        
        ### 🛠️ Technical Details
        
        - **Deep Learning Framework**: TensorFlow 2.x with Keras API
        - **Architecture**: Convolutional Neural Network with batch normalization and dropout
        - **Frontend**: Streamlit with custom CSS styling
        - **Visualization**: Plotly for interactive charts
        - **Image Processing**: PIL and OpenCV for preprocessing
        
        ### 📈 Model Architecture
        
        The CNN model includes:
        - 3 Convolutional blocks with BatchNormalization and Dropout
        - MaxPooling layers for dimensionality reduction
        - Dense layers with regularization
        - Softmax activation for multi-class classification
        
        ### 🎨 UI Features
        
        - Gradient backgrounds and modern card designs
        - Interactive charts and real-time updates
        - Responsive layout for all device sizes
        - Professional color scheme and typography
        """)
        
        # Model summary if available
        if is_trained:
            with st.expander("📋 Model Architecture Summary"):
                st.text(model.get_model_summary())


if __name__ == "__main__":
    main()
