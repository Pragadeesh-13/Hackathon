"""
Dataset utilities for cattle breed recognition.
Handles dataset loading, splitting, and breed discovery.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def discover_breeds(dataset_path: str) -> List[str]:
    """
    Discover all available breeds in the dataset directory.
    
    Args:
        dataset_path: Path to dataset directory containing breed folders
        
    Returns:
        List of breed names (folder names)
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    breeds = []
    for item in dataset_dir.iterdir():
        if item.is_dir():
            # Check if directory contains images
            image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + \
                         list(item.glob("*.png")) + list(item.glob("*.webp"))
            if image_files:
                breeds.append(item.name)
    
    return sorted(breeds)

def get_breed_images(dataset_path: str, breed: str) -> List[str]:
    """
    Get all image paths for a specific breed.
    
    Args:
        dataset_path: Path to dataset directory
        breed: Name of the breed
        
    Returns:
        List of image file paths
    """
    breed_dir = Path(dataset_path) / breed
    if not breed_dir.exists():
        raise ValueError(f"Breed directory {breed_dir} does not exist")
    
    # Supported image extensions
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    
    for ext in extensions:
        image_paths.extend([str(p) for p in breed_dir.glob(ext)])
    
    return sorted(image_paths)

def split_dataset(image_paths: List[str], 
                 train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, 
                 test_ratio: float = 0.15,
                 random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split image paths into train, validation, and test sets.
    
    Args:
        image_paths: List of image file paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    # Ensure ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle the paths
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)
    
    # Calculate split indices
    n_total = len(shuffled_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the data
    train_paths = shuffled_paths[:n_train]
    val_paths = shuffled_paths[n_train:n_train + n_val]
    test_paths = shuffled_paths[n_train + n_val:]
    
    return train_paths, val_paths, test_paths

def get_dataset_splits(dataset_path: str,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15, 
                      test_ratio: float = 0.15,
                      random_seed: int = 42) -> Dict[str, Dict[str, List[str]]]:
    """
    Get train/val/test splits for all breeds in the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with structure:
        {
            'breed_name': {
                'train': [list of train image paths],
                'val': [list of val image paths], 
                'test': [list of test image paths]
            }
        }
    """
    breeds = discover_breeds(dataset_path)
    if not breeds:
        raise ValueError(f"No breeds found in dataset path {dataset_path}")
    
    dataset_splits = {}
    
    for breed in breeds:
        image_paths = get_breed_images(dataset_path, breed)
        if len(image_paths) < 3:  # Need at least 3 images for train/val/test
            print(f"Warning: Breed {breed} has only {len(image_paths)} images, skipping")
            continue
            
        train_paths, val_paths, test_paths = split_dataset(
            image_paths, train_ratio, val_ratio, test_ratio, random_seed
        )
        
        dataset_splits[breed] = {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }
        
        print(f"Breed {breed}: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    return dataset_splits

def get_breed_info() -> Dict[str, Dict[str, str]]:
    """
    Get hardcoded information about Indian cattle and buffalo breeds.
    
    Returns:
        Dictionary with breed information
    """
    breed_info = {
        # Buffalo breeds (5)
        "Murrah": {
            "region": "Punjab, Haryana, Rajasthan",
            "avg_milk_yield": "12-18 liters/day",
            "features": "Large, black body with curled horns. Known for high milk production and docile nature.",
            "type": "Buffalo"
        },
        "Surti": {
            "region": "Gujarat, Maharashtra",
            "avg_milk_yield": "8-12 liters/day",
            "features": "Medium-sized buffalo with light brown to grey color. Known for rich milk with high fat content.",
            "type": "Buffalo"
        },
        "Jaffarbadi": {
            "region": "Gujarat, Rajasthan",
            "avg_milk_yield": "15-20 liters/day",
            "features": "Large, powerful buffalo with distinctive curved horns. Excellent for commercial dairy farming with high milk yield.",
            "type": "Buffalo"
        },
        "Bhadawari": {
            "region": "Uttar Pradesh, Madhya Pradesh",
            "avg_milk_yield": "4-6 liters/day",
            "features": "Medium-sized, hardy buffalo adapted to waterlogged areas. Known for disease resistance and ability to thrive in difficult conditions.",
            "type": "Buffalo"
        },
        "Mehsana": {
            "region": "Gujarat, Rajasthan",
            "avg_milk_yield": "8-12 liters/day",
            "features": "Medium to large buffalo with good heat tolerance. Well-suited for semi-arid regions with consistent milk production.",
            "type": "Buffalo"
        },
        
        # Cattle breeds (5)
        "Gir": {
            "region": "Gujarat, Rajasthan",
            "avg_milk_yield": "8-12 liters/day", 
            "features": "Distinctive hump, drooping ears, and curved horns. Resistant to diseases and heat.",
            "type": "Cattle"
        },
        "Sahiwal": {
            "region": "Punjab, Haryana",
            "avg_milk_yield": "10-15 liters/day",
            "features": "Reddish-brown color with white patches. Good milk production and heat tolerance.",
            "type": "Cattle"
        },
        "Tharparkar": {
            "region": "Rajasthan, Gujarat",
            "avg_milk_yield": "8-10 liters/day",
            "features": "White or light grey color. Dual-purpose breed for milk and draught.",
            "type": "Cattle"
        },
        "Kankrej": {
            "region": "Gujarat, Rajasthan",
            "avg_milk_yield": "6-10 liters/day",
            "features": "Large cattle with silver-grey color. Excellent draught animals with good heat tolerance and disease resistance.",
            "type": "Cattle"
        },
        "Ongole": {
            "region": "Andhra Pradesh, Tamil Nadu",
            "avg_milk_yield": "4-8 liters/day",
            "features": "Large white cattle with prominent hump. Known for heat resistance and strength, primarily used for draught purposes.",
            "type": "Cattle"
        },
        "Red_Sindhi": {
            "region": "Sindh (now Pakistan), Gujarat",
            "avg_milk_yield": "6-10 liters/day",
            "features": "Red colored with white markings. Heat tolerant and disease resistant.",
            "type": "Cattle"
        }
    }
    
    return breed_info