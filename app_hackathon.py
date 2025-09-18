#!/usr/bin/env python3
"""
HACKATHON STREAMLIT APP

Competition-grade web application featuring:
- Ensemble model inference
- Real-time confidence scoring
- Comprehensive breed analysis
- Performance metrics display
"""

import streamlit as st
import sys
import os
import json
import time
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import our hackathon ensemble inference
try:
    from infer_hackathon_ensemble import HackathonEnsembleInference
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    st.error("❌ Hackathon ensemble system not available. Please run train_hackathon_pipeline.py first.")

# Page configuration
st.set_page_config(
    page_title="🏆 Hackathon Cattle & Buffalo Breed Recognition",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .ensemble-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏆 Hackathon Cattle & Buffalo Breed Recognition</h1>
    <p>Competition-Grade Ensemble AI System | 11 Breeds | 95%+ Target Accuracy</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ensemble_system():
    """Load ensemble inference system"""
    if not ENSEMBLE_AVAILABLE:
        return None
    
    try:
        return HackathonEnsembleInference()
    except Exception as e:
        st.error(f"❌ Failed to load ensemble system: {e}")
        return None

def display_ensemble_info(ensemble):
    """Display ensemble system information"""
    if not ensemble:
        return
    
    st.markdown("### 🤖 Ensemble System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Models", f"{ensemble.num_models}")
    
    with col2:
        st.metric("🧬 Breeds", f"{ensemble.num_classes}")
    
    with col3:
        st.metric("📊 Best Accuracy", f"{ensemble.best_accuracy:.1f}%")
    
    with col4:
        st.metric("⚡ Avg Accuracy", f"{ensemble.average_accuracy:.1f}%")
    
    # Ensemble details
    with st.expander("📈 Individual Model Performance"):
        model_data = []
        for i, acc in enumerate(ensemble.individual_accuracies):
            model_data.append({
                'Model': f'Model {i+1}',
                'Accuracy': acc,
                'Status': '🏆 Excellent' if acc >= 90 else '✅ Good' if acc >= 80 else '⚠️ Fair'
            })
        
        fig = px.bar(
            model_data, 
            x='Model', 
            y='Accuracy',
            title="Individual Model Accuracies",
            color='Accuracy',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_prediction_results(result, image):
    """Display comprehensive prediction results"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)
    
    with col2:
        # Main prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎯 Prediction: {result['predicted_breed']}</h2>
            <h3>Confidence: {result['confidence']:.3f}</h3>
            <p>Inference Time: {result['ensemble_details']['inference_time_ms']:.1f}ms</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator
        confidence_color = "green" if result['confidence'] > 0.8 else "orange" if result['confidence'] > 0.6 else "red"
        st.markdown(f"""
        <div style="background-color: {confidence_color}; height: 20px; border-radius: 10px; 
                    width: {result['confidence']*100}%; margin: 10px 0;"></div>
        """, unsafe_allow_html=True)
    
    # Top 3 predictions
    st.markdown("### 🏅 Top 3 Predictions")
    
    for i, pred in enumerate(result['top_3_predictions']):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        st.markdown(f"""
        <div class="metric-card">
            {medal} <strong>{pred['breed']}</strong>: {pred['confidence']:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    # Ensemble voting details
    st.markdown("### 🗳️ Ensemble Voting Details")
    
    ensemble_details = result['ensemble_details']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="ensemble-info">
            <h4>Voting Configuration</h4>
            <p><strong>Method:</strong> {ensemble_details['voting_method']}</p>
            <p><strong>Models:</strong> {ensemble_details['num_models']}</p>
            <p><strong>Processing Time:</strong> {ensemble_details['inference_time_ms']:.1f}ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Model agreement chart
        model_predictions = ensemble_details['model_predictions']
        agreement_data = {}
        
        for model_pred in model_predictions:
            breed = model_pred['prediction']
            agreement_data[breed] = agreement_data.get(breed, 0) + 1
        
        if len(agreement_data) > 1:
            fig = px.pie(
                values=list(agreement_data.values()),
                names=list(agreement_data.keys()),
                title="Model Agreement"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Individual model predictions
    with st.expander("🔍 Individual Model Predictions"):
        model_df_data = []
        for model_pred in model_predictions:
            model_df_data.append({
                'Model': f"Model {model_pred['model_idx'] + 1}",
                'Training Accuracy': f"{model_pred['accuracy']:.1f}%",
                'Prediction': model_pred['prediction'],
                'Confidence': f"{model_pred['confidence']:.3f}"
            })
        
        st.dataframe(model_df_data, use_container_width=True)
    
    # All breed probabilities
    with st.expander("📊 All Breed Probabilities"):
        all_preds = result['all_predictions']
        
        # Create probability chart
        breeds = list(all_preds.keys())
        probabilities = list(all_preds.values())
        
        fig = px.bar(
            x=probabilities,
            y=breeds,
            orientation='h',
            title="Probability Distribution Across All Breeds",
            labels={'x': 'Probability', 'y': 'Breed'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Load ensemble system
    ensemble = load_ensemble_system()
    
    if not ensemble:
        st.error("🚫 Ensemble system not available. Please run the training pipeline first.")
        st.markdown("""
        ### 🛠️ Setup Instructions:
        1. Run `python train_hackathon_pipeline.py` to train ensemble models
        2. Ensure models are saved in `models/` directory
        3. Restart this application
        """)
        return
    
    # Display system info
    display_ensemble_info(ensemble)
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Configuration")
    
    voting_method = st.sidebar.selectbox(
        "🗳️ Voting Method",
        ["weighted_average", "simple_average", "max_confidence"],
        help="Choose ensemble voting strategy"
    )
    
    st.sidebar.markdown("### 📋 Supported Breeds")
    breeds_text = "\n".join([f"• {breed}" for breed in ensemble.breeds])
    st.sidebar.text(breeds_text)
    
    # Main prediction interface
    st.markdown("### 📸 Upload Image for Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of cattle or buffalo"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Make prediction
            with st.spinner("🔄 Running ensemble inference..."):
                start_time = time.time()
                result = ensemble.ensemble_predict(image, voting_method)
                total_time = time.time() - start_time
            
            # Display results
            st.success(f"✅ Prediction complete in {total_time:.2f}s")
            display_prediction_results(result, image)
            
            # Download results
            st.markdown("### 💾 Export Results")
            
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Prediction JSON",
                data=result_json,
                file_name=f"prediction_{result['predicted_breed']}_{int(time.time())}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
    
    # Batch processing option
    st.markdown("### 📁 Batch Processing")
    
    if st.button("🔄 Process Sample Images"):
        # Find sample images for demonstration
        sample_images = []
        for breed in ensemble.breeds[:3]:
            breed_path = Path(breed)
            if breed_path.exists():
                images = list(breed_path.glob('*.jpg'))[:1]
                sample_images.extend(images)
        
        if sample_images:
            st.write(f"Processing {len(sample_images)} sample images...")
            
            progress_bar = st.progress(0)
            results_container = st.container()
            
            for i, img_path in enumerate(sample_images):
                try:
                    image = Image.open(img_path)
                    result = ensemble.ensemble_predict(image, voting_method)
                    
                    with results_container:
                        st.markdown(f"**{img_path.name}**: {result['predicted_breed']} ({result['confidence']:.3f})")
                    
                    progress_bar.progress((i + 1) / len(sample_images))
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing {img_path}: {e}")
            
            st.success("✅ Batch processing complete!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🏆 Hackathon Cattle & Buffalo Breed Recognition System<br>
        Powered by Ensemble Deep Learning | Competition-Grade Accuracy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()