#!/usr/bin/env python3
"""
Streamlit Web App for Cattle Breed Recognition

This is a user-friendly web interface for cattle breed recognition.
Users can upload images and get breed predictions with confidence scores
and detailed breed information.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pickle
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    FeatureExtractor,
    get_device,
    load_and_preprocess_image,
    compute_similarities,
    get_top_predictions,
    get_breed_info
)

# Page configuration
st.set_page_config(
    page_title="Cattle Breed Recognition",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_prototypes(prototypes_path="models/prototypes_maximum_10breed.pkl"):
    """Load maximum 10-breed model and prototypes (cached for performance)."""
    try:
        # Try maximum 10-breed prototypes first
        if not os.path.exists(prototypes_path):
            # Fallback hierarchy
            fallback_paths = [
                "models/prototypes_5breed_ultra.pkl",
                "models/prototypes_5breed_optimized.pkl",
                "models/prototypes_enhanced.pkl",
                "models/prototypes.pkl"
            ]
            
            for fallback_path in fallback_paths:
                if os.path.exists(fallback_path):
                    prototypes_path = fallback_path
                    break
            
            if prototypes_path == "models/prototypes_maximum_10breed.pkl":
                st.sidebar.error("No model found! Please run prototype builder.")
                return None, None, None, None, None, None
            else:
                st.sidebar.warning(f"Using fallback model: {os.path.basename(prototypes_path)}")
        else:
            st.sidebar.success("🚀 Using MAXIMUM 10-BREED model with near-perfect accuracy!")
        
        # Load prototypes
        with open(prototypes_path, 'rb') as f:
            prototype_data = pickle.load(f)
        
        prototypes = prototype_data['prototypes']
        model_name = prototype_data['model_name']
        breeds = prototype_data['breeds']
        config = prototype_data.get('config', {})
        
        # Show optimization level in sidebar
        if config.get('optimization_level') == 'maximum_discrimination_10breed':
            st.sidebar.markdown("### 🔥 Maximum 10-Breed Features:")
            if config.get('near_perfect_targeting'):
                st.sidebar.markdown("✅ Near-Perfect Accuracy Targeting")
            if config.get('cross_species_discrimination'):
                st.sidebar.markdown("✅ Cross-Species Enhancement")
            if config.get('maximum_contrastive_refinement'):
                st.sidebar.markdown("✅ Maximum Contrastive Refinement")
            if config.get('adaptive_feature_weighting'):
                st.sidebar.markdown("✅ Adaptive Feature Weighting")
            if 'total_features' in prototype_data:
                st.sidebar.markdown(f"🎯 Training Features: {prototype_data['total_features']:,}")
        elif config.get('optimization_level') == 'ultra_maximum_differentiation':
            st.sidebar.markdown("### 🔥 Ultra Optimization Features:")
            if config.get('contrastive_refinement'):
                st.sidebar.markdown("✅ Contrastive Refinement")
            if config.get('feature_importance_weighting'):
                st.sidebar.markdown("✅ Feature Importance Weighting") 
            if config.get('ultra_augmentation'):
                st.sidebar.markdown("✅ Ultra Augmentation")
        elif config.get('enhanced_augment'):
            st.sidebar.markdown("### ✨ Enhanced Model Features:")
            st.sidebar.markdown("✅ Enhanced Augmentation")
            if config.get('use_ensemble'):
                st.sidebar.markdown("✅ Ensemble Prototypes")
        
        # Load model
        device = get_device()
        model = FeatureExtractor(model_name=model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        
        return model, prototypes, model_name, breeds, device, config
        
    except Exception as e:
        st.error(f"Error loading model and prototypes: {e}")
        return None, None, None, None, None

def predict_breed(image, model, prototypes, device):
    """Predict breed from uploaded image."""
    try:
        # Preprocess image
        image_tensor = load_and_preprocess_image(image) if isinstance(image, str) else None
        
        if image_tensor is None:
            # Handle PIL Image
            from utils.model import get_transforms
            transform = get_transforms(augment=False)
            image_tensor = transform(image)
        
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(image_batch)
            features_normalized = F.normalize(features, p=2, dim=1)
            query_embedding = features_normalized.squeeze(0)
        
        # Compute similarities
        similarities = compute_similarities(query_embedding, prototypes)
        
        # Get top predictions
        top_predictions = get_top_predictions(similarities, top_k=len(prototypes))
        
        return top_predictions, similarities
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def display_breed_info(breed_name):
    """Display detailed breed information."""
    breed_info = get_breed_info()
    
    if breed_name in breed_info:
        info = breed_info[breed_name]
        
        st.subheader(f"About {breed_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Type", info['type'])
            st.metric("Region", info['region'])
        
        with col2:
            st.metric("Avg. Milk Yield", info['avg_milk_yield'])
        
        st.write("**Characteristics:**")
        st.write(info['features'])
        
    else:
        st.warning(f"No detailed information available for {breed_name}")

def create_confidence_chart(predictions):
    """Create a confidence chart using Plotly."""
    breeds = [pred[0] for pred in predictions]
    scores = [pred[1] for pred in predictions]
    
    # Create color scale (green for high confidence, red for low)
    colors = ['#d73027' if score < 0.4 else '#fee08b' if score < 0.6 else '#66bd63' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=breeds,
            y=scores,
            marker_color=colors,
            text=[f'{score:.3f}' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Breed Prediction Confidence",
        xaxis_title="Breed",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    return fig

def main():
    # Title and description
    st.title("🐄 Indian Cattle & Buffalo Breed Recognition")
    st.markdown("""
    Upload an image of cattle or buffalo to identify the breed using deep learning.
    This system uses maximum 10-breed optimization with near-perfect accuracy for:
    
    **🐃 Buffalo Breeds**: Bhadawari, Jaffarbadi, Mehsana, Murrah, Surti  
    **🐄 Cattle Breeds**: Gir, Kankrej, Ongole, Sahiwal, Tharparkar
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        prototypes_path = st.text_input(
            "Prototypes Path",
            value="models/prototypes_maximum_10breed.pkl",
            help="Path to the saved prototypes file"
        )
        
        show_top_k = st.slider(
            "Show Top Predictions",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of top predictions to display"
        )
        
        show_breed_info = st.checkbox(
            "Show Breed Information",
            value=True,
            help="Display detailed information about the predicted breed"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Minimum confidence score for positive identification. Below this threshold, the image will be classified as 'Unknown' or 'Not recognized breed'."
        )
    
    # Load model and prototypes
    model, prototypes, model_name, breeds, device, config = load_model_and_prototypes(prototypes_path)
    
    if model is None:
        st.error("Failed to load model and prototypes. Please check the prototypes path.")
        st.stop()
    
    # Display model info
    st.success(f"✅ Model loaded: {model_name} with {len(breeds)} breeds")
    st.info(f"Available breeds: {', '.join(breeds)}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload an image of cattle or buffalo (JPG, JPEG, PNG, WebP formats supported)"
    )
    
    if uploaded_file is not None:
        # Validate file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File too large! Please upload an image smaller than 10MB.")
            st.stop()
        
        # Validate file type
        if uploaded_file.type not in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']:
            st.error(f"Unsupported file type: {uploaded_file.type}. Please upload JPG, JPEG, PNG, or WebP.")
            st.stop()
        
        # Display uploaded image
        try:
            # Reset file pointer to beginning for Streamlit UploadedFile
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.error("Please make sure you uploaded a valid image file (JPG, JPEG, PNG, WebP)")
            st.stop()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Input Image", use_container_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        
        with col2:
            st.subheader("Prediction Results")
            
            with st.spinner("Analyzing image..."):
                predictions, similarities = predict_breed(image, model, prototypes, device)
            
            if predictions:
                # Display top prediction prominently
                best_breed, best_score = predictions[0]
                
                # Check if prediction meets threshold
                if best_score >= confidence_threshold:
                    # Positive identification
                    if best_score >= 0.8:
                        confidence_emoji = "🟢"
                        confidence_text = "High Confidence"
                    elif best_score >= 0.7:
                        confidence_emoji = "🟡"
                        confidence_text = "Medium Confidence"
                    else:
                        confidence_emoji = "🟠"
                        confidence_text = "Low Confidence"
                    
                    st.metric(
                        label="✅ Positive Identification",
                        value=best_breed,
                        delta=f"{best_score:.3f} confidence"
                    )
                    
                    st.write(f"{confidence_emoji} **{confidence_text}**")
                    
                    # Show breed info
                    identified_breed = best_breed
                else:
                    # Below threshold - no confident match
                    confidence_emoji = "❌"
                    confidence_text = "No Confident Match"
                    
                    st.metric(
                        label="❌ Classification Result",
                        value="Unknown / Not Recognized",
                        delta=f"Best match: {best_breed} ({best_score:.3f})"
                    )
                    
                    st.write(f"{confidence_emoji} **{confidence_text}**")
                    st.write(f"The image does not confidently match any known breed in the database.")
                    
                    identified_breed = None
        
        # Detailed results
        st.subheader("Detailed Predictions")
        
        # Show top-k predictions in a table
        top_k_predictions = predictions[:show_top_k]
        
        # Create DataFrame for better display
        import pandas as pd
        df = pd.DataFrame(top_k_predictions, columns=['Breed', 'Confidence'])
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.3f}")
        df.index = range(1, len(df) + 1)
        
        st.dataframe(df, use_container_width=True)
        
        # Confidence chart
        st.subheader("Confidence Visualization")
        fig = create_confidence_chart(top_k_predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        # Breed information
        if show_breed_info and identified_breed:
            display_breed_info(identified_breed)
        elif show_breed_info and not identified_breed:
            st.info("ℹ️ No breed information available - image did not match any known breed with sufficient confidence.")
        
        # Technical details (collapsible)
        with st.expander("Technical Details"):
            st.write(f"**Model Architecture:** {model_name}")
            st.write(f"**Feature Extraction:** Pretrained CNN features")
            st.write(f"**Similarity Metric:** Cosine similarity")
            st.write(f"**Total Breeds in Database:** {len(breeds)}")
            
            # Show all similarities
            st.write("**All Breed Similarities:**")
            all_similarities = [(breed, score) for breed, score in similarities.items()]
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            
            df_all = pd.DataFrame(all_similarities, columns=['Breed', 'Similarity'])
            df_all['Similarity'] = df_all['Similarity'].apply(lambda x: f"{x:.3f}")
            df_all.index = range(1, len(df_all) + 1)
            
            st.dataframe(df_all, use_container_width=True)
    
    else:
        st.info("👆 Please upload an image to start breed recognition")
        
        # Show example images or breed gallery
        st.subheader("Supported Breeds")
        breed_info = get_breed_info()
        
        cols = st.columns(min(3, len(breeds)))
        for i, breed in enumerate(breeds):
            with cols[i % 3]:
                st.write(f"**{breed}**")
                if breed in breed_info:
                    st.write(f"Type: {breed_info[breed]['type']}")
                    st.write(f"Region: {breed_info[breed]['region']}")
                else:
                    st.write("Indigenous breed")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔬 Powered by PyTorch & Deep Learning | 📊 Built with Streamlit</p>
        <p><small>For research and educational purposes</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()