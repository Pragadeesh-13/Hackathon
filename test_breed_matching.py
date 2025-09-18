#!/usr/bin/env python3
"""
Test script to check breed name matching
"""

# Test breed matching issue
cnn_similarities = {
    'Bhadawari': 0.754365,
    'Gir': 0.813683, 
    'Jaffarbadi': 0.673463,
    'Kankrej': 0.818147,
    'Mehsana': 0.711032,
    'Murrah': 0.689854,
    'Ongole': 0.693313,
    'Sahiwal': 0.790749,
    'Surti': 0.811480,
    'Tharparkar': 0.699605
}

breeds = ['bhadawari', 'gir', 'jaffarbadi', 'kankrej', 'mehsana', 'murrah', 'ongole', 'sahiwal', 'surti', 'tharparkar']

print("CNN similarities keys:", list(cnn_similarities.keys()))
print("Breeds list:", breeds)

# Check if breed matching works
for breed in breeds:
    cnn_score = cnn_similarities.get(breed, 0.0)
    print(f"{breed}: {cnn_score}")

print("\nFixed matching:")
for breed in breeds:
    # Fix: match by capitalizing the breed name
    cnn_score = cnn_similarities.get(breed.capitalize(), 0.0)
    print(f"{breed}: {cnn_score}")