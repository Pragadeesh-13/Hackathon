"""
Model utilities for cattle breed recognition.
Handles model loading, feature extraction, and similarity computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

class FeatureExtractor(nn.Module):
    """
    Feature extractor using pretrained models.
    Removes the final classification layer to get embeddings.
    """
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
            
        elif model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1280
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        """Extract features from input images."""
        features = self.backbone(x)
        # Flatten if needed (for ResNet)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return features

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Get image transforms for preprocessing.
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    if augment:
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)

def load_and_preprocess_image(image_path: str, transform: Optional[transforms.Compose] = None) -> torch.Tensor:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        transform: Optional transform to apply
        
    Returns:
        Preprocessed image tensor
    """
    if transform is None:
        transform = get_transforms(augment=False)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image)
    
    return image_tensor

def extract_features(model: FeatureExtractor, 
                    image_paths: List[str], 
                    device: torch.device,
                    batch_size: int = 32,
                    augment: bool = False) -> torch.Tensor:
    """
    Extract features from a list of images.
    
    Args:
        model: Feature extraction model
        image_paths: List of image file paths
        device: Device to run inference on
        batch_size: Batch size for processing
        augment: Whether to apply augmentation
        
    Returns:
        Feature tensor of shape (num_images, feature_dim)
    """
    model.eval()
    transform = get_transforms(augment=augment)
    
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image_tensor = load_and_preprocess_image(path, transform)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # Stack images into batch
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Extract features
            features = model(batch_tensor)
            all_features.append(features.cpu())
    
    if not all_features:
        raise ValueError("No features extracted. Check your image paths and formats.")
    
    return torch.cat(all_features, dim=0)

def compute_prototypes(features_by_breed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute prototype vectors for each breed.
    
    Args:
        features_by_breed: Dictionary mapping breed names to feature tensors
        
    Returns:
        Dictionary mapping breed names to prototype vectors
    """
    prototypes = {}
    
    for breed, features in features_by_breed.items():
        # Compute mean embedding
        prototype = torch.mean(features, dim=0)
        
        # L2 normalize
        prototype = F.normalize(prototype, p=2, dim=0)
        
        prototypes[breed] = prototype
        print(f"Computed prototype for {breed}: {features.shape[0]} samples -> {prototype.shape}")
    
    return prototypes

def compute_similarities(query_embedding: torch.Tensor, 
                        prototypes: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute cosine similarities between query embedding and all prototypes.
    
    Args:
        query_embedding: Normalized query embedding vector
        prototypes: Dictionary of breed prototypes
        
    Returns:
        Dictionary mapping breed names to similarity scores
    """
    similarities = {}
    
    # Ensure query embedding is normalized
    query_embedding = F.normalize(query_embedding, p=2, dim=0)
    
    for breed, prototype in prototypes.items():
        # Compute cosine similarity
        similarity = torch.dot(query_embedding, prototype).item()
        similarities[breed] = similarity
    
    return similarities

def get_top_predictions(similarities: Dict[str, float], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Get top-k predictions based on similarity scores.
    
    Args:
        similarities: Dictionary mapping breed names to similarity scores
        top_k: Number of top predictions to return
        
    Returns:
        List of (breed_name, similarity_score) tuples, sorted by similarity
    """
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_k]

def print_predictions(predictions: List[Tuple[str, float]], title: str = "Top predictions:"):
    """
    Print formatted predictions.
    
    Args:
        predictions: List of (breed_name, similarity_score) tuples
        title: Title to print above predictions
    """
    print(f"\n{title}")
    for breed, score in predictions:
        print(f"  {breed} ({score:.3f})")
    
    if predictions:
        best_breed, best_score = predictions[0]
        print(f"\nPREDICTED: {best_breed} (similarity={best_score:.3f})")