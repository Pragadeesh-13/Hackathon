#!/usr/bin/env python3
"""
Enhanced Feature Extractor for Hackathon Training

Supports multiple CNN architectures with flexible feature extraction
and classification heads optimized for competition performance.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Union, List
import numpy as np
from PIL import Image

class FeatureExtractor(nn.Module):
    """
    Enhanced feature extractor supporting multiple architectures
    with competition-grade performance optimizations.
    """
    
    def __init__(self, 
                 model_name: str = 'resnet50', 
                 pretrained: bool = True,
                 num_classes: Optional[int] = None,
                 dropout_rate: float = 0.5):
        """
        Initialize feature extractor
        
        Args:
            model_name: Architecture name (resnet50, resnet101, efficientnet_b0, etc.)
            pretrained: Use pretrained weights
            num_classes: Number of output classes (None to keep original)
            dropout_rate: Dropout rate for classification head
        """
        super(FeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load base model
        self.model = self._create_model(model_name, pretrained)
        
        # Modify classifier if num_classes specified
        if num_classes is not None:
            self._modify_classifier(num_classes, dropout_rate)
        
        # Store feature dimensions
        self.feature_dim = self._get_feature_dimension()
    
    def _create_model(self, model_name: str, pretrained: bool):
        """Create the base model architecture"""
        
        model_name = model_name.lower()
        
        if model_name == 'resnet50':
            return models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            return models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            return models.resnet152(pretrained=pretrained)
        elif model_name == 'efficientnet_b0':
            return models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b4':
            return models.efficientnet_b4(pretrained=pretrained)
        elif model_name == 'vgg16':
            return models.vgg16(pretrained=pretrained)
        elif model_name == 'densenet121':
            return models.densenet121(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _modify_classifier(self, num_classes: int, dropout_rate: float):
        """Modify the classifier head for custom number of classes"""
        
        if hasattr(self.model, 'fc'):
            # ResNet, EfficientNet
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(512, num_classes)
            )
        elif hasattr(self.model, 'classifier'):
            # VGG, DenseNet
            if isinstance(self.model.classifier, nn.Sequential):
                # VGG
                in_features = self.model.classifier[-1].in_features
                self.model.classifier = nn.Sequential(
                    *list(self.model.classifier.children())[:-1],
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
            else:
                # DenseNet
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate * 0.5),
                    nn.Linear(512, num_classes)
                )
    
    def _get_feature_dimension(self):
        """Get the dimension of extracted features"""
        
        if hasattr(self.model, 'fc'):
            if isinstance(self.model.fc, nn.Sequential):
                # Find the first Linear layer in the sequential
                for layer in self.model.fc:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
                return 2048  # Default for ResNet50
            else:
                return self.model.fc.in_features
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                for layer in self.model.classifier:
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            else:
                return self.model.classifier.in_features
        
        return 1000  # Default
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def extract_features(self, x, layer_name: Optional[str] = None):
        """
        Extract features from intermediate layers
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract from (None for final features)
        
        Returns:
            Feature tensor
        """
        
        if layer_name is None:
            # Extract final features before classifier
            if hasattr(self.model, 'fc'):
                # ResNet-style
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                
                return x
            else:
                # Use model as feature extractor
                features = self.model.features(x) if hasattr(self.model, 'features') else x
                return torch.flatten(features, 1)
        else:
            # Extract from specific layer (implementation depends on architecture)
            return self._extract_layer_features(x, layer_name)
    
    def _extract_layer_features(self, x, layer_name: str):
        """Extract features from a specific layer"""
        
        # This is a simplified implementation
        # In practice, you'd need to register forward hooks
        features = {}
        
        def hook_fn(module, input, output):
            features[layer_name] = output
        
        # Register hook (simplified - would need proper layer mapping)
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        _ = self.forward(x)
        
        if hasattr(locals(), 'handle'):
            handle.remove()
        
        return features.get(layer_name, x)
    
    def get_model_summary(self):
        """Get model architecture summary"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dropout_rate': self.dropout_rate
        }

def get_device():
    """Get the best available device for computation"""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

def get_transform(image_size: int = 224, 
                 augment: bool = False,
                 mean: List[float] = None,
                 std: List[float] = None):
    """
    Get image transformation pipeline
    
    Args:
        image_size: Target image size
        augment: Apply data augmentation
        mean: Normalization mean (defaults to ImageNet)
        std: Normalization std (defaults to ImageNet)
    
    Returns:
        Transform pipeline
    """
    
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def load_pretrained_model(model_path: str, 
                         model_name: str = 'resnet50',
                         num_classes: int = 1000,
                         device: Optional[torch.device] = None):
    """
    Load a pretrained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_name: Model architecture name
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    
    if device is None:
        device = get_device()
    
    # Create model
    model = FeatureExtractor(model_name, pretrained=False, num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def test_model_architecture():
    """Test different model architectures"""
    
    print("🧪 Testing Model Architectures")
    print("=" * 50)
    
    architectures = ['resnet50', 'resnet101', 'efficientnet_b0']
    
    for arch in architectures:
        try:
            model = FeatureExtractor(arch, pretrained=False, num_classes=11)
            summary = model.get_model_summary()
            
            print(f"✅ {arch}:")
            print(f"   📊 Parameters: {summary['total_params']:,}")
            print(f"   🎯 Feature Dim: {summary['feature_dim']}")
            print(f"   🧬 Classes: {summary['num_classes']}")
            
        except Exception as e:
            print(f"❌ {arch}: {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_model_architecture()