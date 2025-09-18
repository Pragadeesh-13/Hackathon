"""
Utilities package for cattle breed recognition.
"""

from .dataset import (
    discover_breeds,
    get_breed_images, 
    split_dataset,
    get_dataset_splits,
    get_breed_info
)

from .model import (
    FeatureExtractor,
    get_device,
    get_transforms,
    load_and_preprocess_image,
    extract_features,
    compute_prototypes,
    compute_similarities,
    get_top_predictions,
    print_predictions
)

__all__ = [
    "discover_breeds",
    "get_breed_images",
    "split_dataset", 
    "get_dataset_splits",
    "get_breed_info",
    "FeatureExtractor",
    "get_device",
    "get_transforms",
    "load_and_preprocess_image", 
    "extract_features",
    "compute_prototypes",
    "compute_similarities",
    "get_top_predictions",
    "print_predictions"
]