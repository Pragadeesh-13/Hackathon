#!/usr/bin/env python3
"""
GradCAM Visualization for Cattle Breed Recognition

This script generates Grad-CAM visualizations to show which parts of an image
the model focuses on when making breed predictions.

Usage:
    python gradcam.py <image_path> [--prototypes-path path] [--output-dir path]
"""

import argparse
import pickle
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils import (
    FeatureExtractor,
    get_device,
    get_transforms,
    compute_similarities,
    get_top_predictions
)

class GradCAM:
    """Grad-CAM implementation for feature extractors."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activations during forward pass."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index (not used for feature extraction)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        model_output = self.model(input_tensor)
        
        # For feature extractors, we'll use the mean of the output as target
        if class_idx is None:
            target = torch.mean(model_output)
        else:
            target = model_output[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

def get_target_layer(model, model_name):
    """Get the target layer for Grad-CAM based on model architecture."""
    if model_name == "resnet50":
        # For ResNet50, use the last convolutional layer before avgpool
        return model.backbone[-3][-1].conv3  # Last bottleneck's conv3
    elif model_name == "efficientnet_b0":
        # For EfficientNet, use the last convolutional layer
        return model.backbone.features[-1][0]  # Last inverted residual block
    else:
        raise ValueError(f"Unsupported model for GradCAM: {model_name}")

def overlay_heatmap(image, heatmap, alpha=0.6):
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image as numpy array (H, W, C)
        heatmap: Heatmap as numpy array (H, W)
        alpha: Transparency factor
        
    Returns:
        Overlayed image
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to color (using jet colormap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = image * (1 - alpha) + heatmap_colored * alpha
    
    return overlayed.astype(np.uint8)

def load_prototypes(prototypes_path: str):
    """Load prototypes from pickle file."""
    if not os.path.exists(prototypes_path):
        raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")
    
    with open(prototypes_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("image_path",
                       help="Path to image file")
    parser.add_argument("--prototypes-path",
                       default="models/prototypes.pkl",
                       help="Path to prototypes file")
    parser.add_argument("--output-dir",
                       default="models/gradcam_outputs",
                       help="Directory to save visualizations")
    parser.add_argument("--top-breeds",
                       type=int,
                       default=3,
                       help="Number of top breeds to visualize")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CATTLE BREED RECOGNITION - GRAD-CAM VISUALIZATION")
    print("=" * 60)
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prototypes
    print(f"Loading prototypes from '{args.prototypes_path}'...")
    try:
        prototype_data = load_prototypes(args.prototypes_path)
        prototypes = prototype_data['prototypes']
        model_name = prototype_data['model_name']
        breeds = prototype_data['breeds']
        
        print(f"Loaded prototypes for {len(breeds)} breeds using {model_name}")
        
    except Exception as e:
        print(f"Error loading prototypes: {e}")
        return
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"\nLoading {model_name} model...")
    model = FeatureExtractor(model_name=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Get target layer for Grad-CAM
    try:
        target_layer = get_target_layer(model, model_name)
        print(f"Using target layer: {target_layer}")
    except Exception as e:
        print(f"Error getting target layer: {e}")
        return
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    print(f"\nProcessing image: {args.image_path}")
    
    # Load original image for visualization
    original_image = Image.open(args.image_path).convert('RGB')
    original_array = np.array(original_image)
    
    # Preprocess for model
    transform = get_transforms(augment=False)
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get predictions first
    print("Getting predictions...")
    with torch.no_grad():
        features = model(input_tensor)
        features_normalized = F.normalize(features, p=2, dim=1)
        query_embedding = features_normalized.squeeze(0)
    
    similarities = compute_similarities(query_embedding, prototypes)
    top_predictions = get_top_predictions(similarities, top_k=args.top_breeds)
    
    print(f"\nTop {args.top_breeds} predictions:")
    for breed, score in top_predictions:
        print(f"  {breed}: {score:.3f}")
    
    # Generate Grad-CAM for top predictions
    print(f"\nGenerating Grad-CAM visualizations...")
    
    # Create figure
    fig, axes = plt.subplots(2, args.top_breeds + 1, figsize=(4 * (args.top_breeds + 1), 8))
    if args.top_breeds == 1:
        axes = axes.reshape(2, -1)
    
    # Show original image
    axes[0, 0].imshow(original_array)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_array)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis('off')
    
    for i, (breed, score) in enumerate(top_predictions):
        print(f"Generating Grad-CAM for {breed}...")
        
        try:
            # Generate Grad-CAM
            # Note: For feature extractors, we'll use a generic approach
            cam = gradcam.generate_cam(input_tensor)
            
            # Overlay heatmap
            overlayed = overlay_heatmap(original_array, cam, alpha=0.6)
            
            # Plot heatmap
            axes[0, i + 1].imshow(cam, cmap='jet')
            axes[0, i + 1].set_title(f'{breed}\nHeatmap ({score:.3f})')
            axes[0, i + 1].axis('off')
            
            # Plot overlay
            axes[1, i + 1].imshow(overlayed)
            axes[1, i + 1].set_title(f'{breed}\nOverlay ({score:.3f})')
            axes[1, i + 1].axis('off')
            
            # Save individual heatmap
            heatmap_path = os.path.join(args.output_dir, f'gradcam_{breed}_{score:.3f}.png')
            plt.imsave(heatmap_path, cam, cmap='jet')
            
            # Save overlay
            overlay_path = os.path.join(args.output_dir, f'overlay_{breed}_{score:.3f}.png')
            plt.imsave(overlay_path, overlayed)
            
        except Exception as e:
            print(f"Error generating Grad-CAM for {breed}: {e}")
            # Show empty plots
            axes[0, i + 1].text(0.5, 0.5, f'Error\n{breed}', ha='center', va='center')
            axes[0, i + 1].axis('off')
            axes[1, i + 1].text(0.5, 0.5, f'Error\n{breed}', ha='center', va='center')
            axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    
    # Save complete visualization
    output_path = os.path.join(args.output_dir, 'gradcam_complete.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComplete visualization saved to: {output_path}")
    
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("GRAD-CAM VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Files generated:")
    print(f"  - gradcam_complete.png (complete visualization)")
    for breed, score in top_predictions:
        print(f"  - gradcam_{breed}_{score:.3f}.png (heatmap)")
        print(f"  - overlay_{breed}_{score:.3f}.png (overlay)")

if __name__ == "__main__":
    main()