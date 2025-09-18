#!/usr/bin/env python3
"""
Enhanced Streamlit App with 11-Breed Support Including Nagpuri

This app showcases the complete breed recognition system with:
- Enhanced 11-breed trained model including Nagpuri
- Smart CNN-first fusion with morphological tiebreaking
- Real-time predictions with confidence scores
- Detailed breed information and insights
"""

import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import pickle
import json
from io import BytesIO
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add utils to path
sys.path.append('.')
from utils import get_device, get_transforms
from utils.feature_extractor import FeatureExtractor  # Use the full feature extractor
from utils.smart_fusion import SmartCNNFirstFusion

# Configure Streamlit page
st.set_page_config(
    page_title="🧬 Enhanced 11-Breed Cattle & Buffalo Recognition",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_enhanced_model():
    """Load the enhanced 11-breed model and system"""
    # Load model info
    with open('models/enhanced_11breed_info.json', 'r') as f:
        model_info = json.load(f)
    
    # Load trained model
    device = get_device()
    model = FeatureExtractor(model_name='resnet50', num_classes=11)
    model.load_state_dict(torch.load('models/enhanced_11breed_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load prototypes for smart fusion
    with open('models/prototypes_maximum_11breed.pkl', 'rb') as f:
        prototype_data = pickle.load(f)
    
    # Initialize smart fusion system
    fusion_system = SmartCNNFirstFusion(model_info['breeds'])
    
    return model, model_info, prototype_data, fusion_system, device

def classify_with_trained_model(image, model, model_info, device):
    """Classify image using the trained 11-breed model"""
    transform = get_transforms(augment=False)
    
    # Preprocess image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences = probabilities[0].cpu().numpy()
    
    # Create breed predictions
    predictions = []
    for i, breed in enumerate(model_info['breeds']):
        predictions.append({
            'breed': breed,
            'confidence': float(confidences[i]),
            'percentage': float(confidences[i] * 100)
        })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def smart_fusion_predict(image, model_info, prototype_data, fusion_system, device):
    """Perform smart fusion prediction"""
    try:
        # Load prototypes to device
        prototypes = {}
        for breed in model_info['breeds']:
            if breed in prototype_data['prototypes']:
                prototypes[breed] = prototype_data['prototypes'][breed].to(device)
        
        # Create model for feature extraction (use the basic one for prototypes)
        from utils import FeatureExtractor as BasicFeatureExtractor
        feature_model = BasicFeatureExtractor(model_name='resnet50', pretrained=True)
        feature_model = feature_model.to(device)
        feature_model.eval()
        
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Perform smart fusion prediction
        predictions, smart_prediction = fusion_system.predict_with_smart_fusion(
            image, feature_model, prototypes, device
        )
        
        return predictions, smart_prediction
    except Exception as e:
        st.error(f"Smart fusion error: {e}")
        return None, None

def get_breed_details(breed_name):
    """Get detailed information about a breed"""
    breed_info = {
        "Bhadawari": {
            "category": "Buffalo",
            "origin": "Uttar Pradesh & Madhya Pradesh, India",
            "characteristics": "Medium-sized, greyish-black color, prominent horns",
            "milk_yield": "1000-1200 liters per lactation",
            "specialty": "Hardy breed, good for milk and draft work"
        },
        "Gir": {
            "category": "Cattle (Zebu)",
            "origin": "Gujarat, India",
            "characteristics": "White/red patches, distinctive lyre-shaped horns",
            "milk_yield": "1200-1800 liters per lactation",
            "specialty": "Excellent milk producer, heat tolerant"
        },
        "Jaffarbadi": {
            "category": "Buffalo",
            "origin": "Gujarat, India",
            "characteristics": "Large size, black color, massive body",
            "milk_yield": "2000-2500 liters per lactation",
            "specialty": "One of the heaviest buffalo breeds"
        },
        "Kankrej": {
            "category": "Cattle (Zebu)",
            "origin": "Gujarat & Rajasthan, India",
            "characteristics": "Silver-grey color, lyre-shaped horns",
            "milk_yield": "1400-2000 liters per lactation",
            "specialty": "Dual-purpose: milk and draft"
        },
        "Mehsana": {
            "category": "Buffalo",
            "origin": "Gujarat, India",
            "characteristics": "Medium to large size, black coat",
            "milk_yield": "1800-2200 liters per lactation",
            "specialty": "High milk fat content"
        },
        "Murrah": {
            "category": "Buffalo",
            "origin": "Haryana & Punjab, India",
            "characteristics": "Black color, tightly curled horns",
            "milk_yield": "2000-3000 liters per lactation",
            "specialty": "World's best dairy buffalo breed"
        },
        "Nagpuri": {
            "category": "Buffalo",
            "origin": "Maharashtra, India",
            "characteristics": "Medium size, black color, good body conformation",
            "milk_yield": "1500-2000 liters per lactation",
            "specialty": "Hardy breed adapted to local conditions"
        },
        "Ongole": {
            "category": "Cattle (Zebu)",
            "origin": "Andhra Pradesh, India",
            "characteristics": "White/light grey, large size, short horns",
            "milk_yield": "1000-1500 liters per lactation",
            "specialty": "Heat resistant, good for beef"
        },
        "Sahiwal": {
            "category": "Cattle (Zebu)",
            "origin": "Punjab (Pakistan/India)",
            "characteristics": "Red/brown color, drooping ears",
            "milk_yield": "2000-3000 liters per lactation",
            "specialty": "Excellent milk producer"
        },
        "Surti": {
            "category": "Buffalo",
            "origin": "Gujarat, India",
            "characteristics": "Medium size, distinct white markings",
            "milk_yield": "1500-2000 liters per lactation",
            "specialty": "Good milk quality with high fat"
        },
        "Tharparkar": {
            "category": "Cattle (Zebu)",
            "origin": "Rajasthan, India",
            "characteristics": "White/grey color, medium size",
            "milk_yield": "1500-2500 liters per lactation",
            "specialty": "Heat and drought tolerant"
        }
    }
    
    return breed_info.get(breed_name, {
        "category": "Unknown",
        "origin": "Information not available",
        "characteristics": "Details not available",
        "milk_yield": "Data not available",
        "specialty": "Information not available"
    })

def main():
    st.title("🧬 Enhanced 11-Breed Cattle & Buffalo Recognition")
    st.markdown("**🚀 Advanced AI System with Trained Model + Smart Fusion + Nagpuri Support**")
    
    # Load system
    with st.spinner("Loading enhanced AI system..."):
        model, model_info, prototype_data, fusion_system, device = load_enhanced_model()
    
    # Sidebar info
    with st.sidebar:
        st.header("🔧 System Information")
        st.success(f"**Model Accuracy: {model_info['best_accuracy']:.1f}%**")
        
        st.info("""
        **Enhanced Features:**
        
        🧠 **Trained CNN**: ResNet50 with 11-breed classification
        
        🔄 **Smart Fusion**: CNN-first with morphological tiebreaking
        
        🆕 **Nagpuri Support**: Newly integrated breed
        
        📊 **Real-time**: Instant predictions with confidence
        
        🎯 **High Accuracy**: 79.8% overall accuracy
        """)
        
        st.header("🐄 Supported Breeds (11)")
        
        # Show breed accuracies
        st.subheader("🐃 Buffalo Breeds")
        buffalo_breeds = ["Bhadawari", "Jaffarbadi", "Mehsana", "Murrah", "Nagpuri", "Surti"]
        for breed in buffalo_breeds:
            if breed in model_info['breed_accuracies']:
                accuracy = model_info['breed_accuracies'][breed]
                st.write(f"• {breed}: {accuracy:.1f}%")
        
        st.subheader("🐄 Cattle Breeds")
        cattle_breeds = ["Gir", "Kankrej", "Ongole", "Sahiwal", "Tharparkar"]
        for breed in cattle_breeds:
            if breed in model_info['breed_accuracies']:
                accuracy = model_info['breed_accuracies'][breed]
                st.write(f"• {breed}: {accuracy:.1f}%")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image of cattle or buffalo",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image showing the animal's full body"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy for processing
            image_np = np.array(image)
            
            # Analysis options
            st.subheader("🔬 Analysis Options")
            analysis_method = st.radio(
                "Choose analysis method:",
                ["Trained Model Only", "Smart Fusion", "Both Methods"],
                help="Select how you want to analyze the image"
            )
    
    with col2:
        if uploaded_file is not None:
            st.header("🎯 Analysis Results")
            
            with st.spinner("Analyzing image..."):
                
                if analysis_method in ["Trained Model Only", "Both Methods"]:
                    st.subheader("🧠 Trained Model Predictions")
                    
                    # Get trained model predictions
                    trained_predictions = classify_with_trained_model(image_np, model, model_info, device)
                    
                    # Display top prediction
                    top_pred = trained_predictions[0]
                    st.success(f"**🥇 Predicted Breed: {top_pred['breed']}**")
                    st.info(f"**Confidence: {top_pred['percentage']:.1f}%**")
                    
                    # Show top 5 predictions
                    st.subheader("📊 Top 5 Predictions")
                    for i, pred in enumerate(trained_predictions[:5]):
                        emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                        st.write(f"{emoji} {pred['breed']}: {pred['percentage']:.1f}%")
                    
                    # Confidence chart
                    fig = px.bar(
                        x=[p['breed'] for p in trained_predictions[:5]],
                        y=[p['percentage'] for p in trained_predictions[:5]],
                        title="Confidence Scores",
                        labels={'x': 'Breed', 'y': 'Confidence (%)'},
                        color=[p['percentage'] for p in trained_predictions[:5]],
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                if analysis_method in ["Smart Fusion", "Both Methods"]:
                    if analysis_method == "Both Methods":
                        st.markdown("---")
                    
                    st.subheader("🔄 Smart Fusion Analysis")
                    
                    # Get smart fusion predictions
                    fusion_predictions, smart_prediction = smart_fusion_predict(
                        image_np, model_info, prototype_data, fusion_system, device
                    )
                    
                    if smart_prediction:
                        # Display fusion results
                        top_breed = max(smart_prediction.breed_scores.items(), key=lambda x: x[1])
                        st.success(f"**🎯 Smart Fusion Result: {top_breed[0]}**")
                        st.info(f"**Score: {top_breed[1]:.3f}**")
                        st.info(f"**Strategy: {smart_prediction.fusion_strategy}**")
                        st.info(f"**Confidence Level: {smart_prediction.confidence_level}**")
                        
                        # Show explanation
                        st.write(f"💡 **Explanation:** {smart_prediction.explanation}")
                        
                        # Strategy details
                        with st.expander("⚙️ Fusion Strategy Details"):
                            st.write(f"🧠 CNN Confidence: {smart_prediction.cnn_confidence:.3f}")
                            st.write(f"🔬 Morphological Confidence: {smart_prediction.morpho_confidence:.3f}")
                            st.write(f"📏 Margin: {smart_prediction.margin:.3f}")
            
            # Breed Information
            if uploaded_file is not None:
                st.markdown("---")
                st.header("📖 Breed Information")
                
                # Determine which breed to show info for
                if analysis_method == "Trained Model Only":
                    info_breed = trained_predictions[0]['breed']
                elif analysis_method == "Smart Fusion":
                    if smart_prediction:
                        info_breed = max(smart_prediction.breed_scores.items(), key=lambda x: x[1])[0]
                    else:
                        info_breed = None
                else:  # Both methods
                    info_breed = trained_predictions[0]['breed']
                
                if info_breed:
                    breed_details = get_breed_details(info_breed)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**🏷️ Category:** {breed_details['category']}")
                        st.write(f"**🌍 Origin:** {breed_details['origin']}")
                        st.write(f"**🥛 Milk Yield:** {breed_details['milk_yield']}")
                    
                    with col_b:
                        st.write(f"**⭐ Specialty:** {breed_details['specialty']}")
                        st.write(f"**🎨 Characteristics:** {breed_details['characteristics']}")

if __name__ == "__main__":
    # Check if model files exist
    if not os.path.exists('models/enhanced_11breed_model.pth'):
        st.error("❌ Enhanced 11-breed model not found!")
        st.info("Please ensure the trained model file 'enhanced_11breed_model.pth' exists in the models directory.")
    elif not os.path.exists('models/prototypes_maximum_11breed.pkl'):
        st.error("❌ 11-breed prototypes not found!")
        st.info("Please ensure the prototypes file 'prototypes_maximum_11breed.pkl' exists in the models directory.")
    else:
        main()